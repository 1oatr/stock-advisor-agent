"""risk/engine.py — 风控引擎

多级风控拦截，拥有最高优先级，优先于策略执行。
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, date
import pandas as pd


@dataclass
class RiskContext:
    """风控上下文"""
    date: datetime
    cash: float
    total_equity: float
    positions: Dict
    daily_trade_count: int = 0
    daily_loss: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    market_state: str = "unknown"
    volatility: float = 0.0


@dataclass
class RiskResult:
    """风控结果"""
    approved: bool = True
    reject_reason: str = ""
    suggested_action: str = "normal"  # normal / reduce / suspend
    suggested_position_ratio: float = 1.0


class RiskRule:
    """风控规则基类"""

    def check(self, ctx: RiskContext, order=None) -> RiskResult:
        raise NotImplementedError


class PositionLimitRule(RiskRule):
    """单标的风控：最大持仓比例限制"""

    def __init__(self, max_position_ratio: float = 0.2):
        self.max_ratio = max_position_ratio

    def check(self, ctx: RiskContext, order=None) -> RiskResult:
        if order and order["direction"] == "buy":
            pos = ctx.positions.get(order["code"])
            current_ratio = (pos.volume * pos.current_price / ctx.total_equity) if pos else 0
            if current_ratio >= self.max_ratio:
                return RiskResult(approved=False, reject_reason=f"单标持仓超限 {self.max_ratio:.0%}")
        return RiskResult()


class DrawdownLimitRule(RiskRule):
    """整体回撤风控：超限自动暂停开仓/降仓"""

    def __init__(self, daily_loss_limit: float = 0.03, total_drawdown_limit: float = 0.15,
                 suspend_level: float = 0.20):
        self.daily_loss_limit = daily_loss_limit
        self.total_drawdown_limit = total_drawdown_limit
        self.suspend_level = suspend_level
        self.suspended = False

    def check(self, ctx: RiskContext, order=None) -> RiskResult:
        if self.suspended:
            return RiskResult(approved=False, reject_reason="风控冻结中（回撤超限）")

        # 日亏损超限
        if ctx.daily_loss > self.daily_loss_limit * ctx.total_equity:
            if order and order["direction"] == "buy":
                return RiskResult(approved=False, reject_reason=f"当日亏损超限 {self.daily_loss_limit:.0%}")

        # 总回撤超限
        if ctx.current_drawdown > self.total_drawdown_limit:
            if order and order["direction"] == "buy":
                return RiskResult(
                    approved=False,
                    reject_reason=f"回撤超限 {self.total_drawdown_limit:.0%}",
                    suggested_action="reduce",
                    suggested_position_ratio=0.5,
                )

        # 极端回撤，冻结
        if ctx.current_drawdown > self.suspend_level:
            self.suspended = True
            return RiskResult(approved=False, reject_reason="极端回撤，冻结所有开仓")

        return RiskResult()


class MarketRiskRule(RiskRule):
    """市场风控：极端行情自动屏蔽买入"""

    def __init__(self):
        self.bear_mode = False

    def check(self, ctx: RiskContext, order=None) -> RiskResult:
        if ctx.market_state == "bear" and order and order["direction"] == "buy":
            return RiskResult(
                approved=False,
                reject_reason="熊市环境，屏蔽买入信号",
                suggested_action="reduce",
                suggested_position_ratio=0.3,
            )
        # 高波动率环境
        if ctx.volatility > 0.04 and order and order["direction"] == "buy":
            return RiskResult(
                approved=False,
                reject_reason="高波动率环境，暂不开仓",
            )
        return RiskResult()


class FrequencyRiskRule(RiskRule):
    """交易频率风控：限制单日最大交易次数"""

    def __init__(self, max_daily_trades: int = 10):
        self.max_daily = max_daily_trades

    def check(self, ctx: RiskContext, order=None) -> RiskResult:
        if ctx.daily_trade_count >= self.max_daily:
            return RiskResult(approved=False, reject_reason=f"当日交易次数超限 ({self.max_daily})")
        return RiskResult()


class AbnormalFilterRule(RiskRule):
    """异常拦截：价格异动、流动性不足、涨跌停封单"""

    def check(self, ctx: RiskContext, order=None) -> RiskResult:
        if order is None:
            return RiskResult()

        bar = order.get("bar", {})
        # 涨跌停封单
        if bar.get("is_limit_up", False) and order["direction"] == "buy":
            return RiskResult(approved=False, reject_reason="涨停封板，禁止开仓")
        if bar.get("is_limit_down", False) and order["direction"] == "sell":
            return RiskResult(approved=False, reject_reason="跌停封板，禁止卖出")
        # 流动性不足
        if bar.get("volume", 0) < 10000:  # 成交量 < 1万股
            return RiskResult(approved=False, reject_reason="流动性不足")
        return RiskResult()


class RiskEngine:
    """风控引擎 — 组合所有风控规则按顺序执行"""

    def __init__(self):
        self.rules: List[RiskRule] = []
        self._init_default_rules()

    def _init_default_rules(self):
        self.register(PositionLimitRule())
        self.register(DrawdownLimitRule())
        self.register(MarketRiskRule())
        self.register(FrequencyRiskRule())
        self.register(AbnormalFilterRule())

    def register(self, rule: RiskRule):
        """注册自定义风控规则"""
        self.rules.append(rule)

    def check(self, ctx: RiskContext, order: Optional[Dict] = None) -> RiskResult:
        """执行所有风控规则（全部通过才通过）"""
        for rule in self.rules:
            result = rule.check(ctx, order)
            if not result.approved:
                return result
        return RiskResult()

    def check_order(self, order: Dict, ctx: RiskContext) -> RiskResult:
        """检查单个订单（返回是否允许执行）"""
        return self.check(ctx, order)

    def check_open_position(self, ctx: RiskContext) -> bool:
        """是否允许开新仓"""
        result = self.check(ctx)
        return result.approved
