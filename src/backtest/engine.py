"""backtest/engine.py — 事件驱动回测引擎

核心回测循环：数据输入 → 信号生成 → 风控校验 → 执行 → 绩效统计
"""

from typing import List, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np


class OrderDirection(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    """订单"""
    code: str
    direction: OrderDirection
    price: float
    volume: int
    order_type: OrderType = OrderType.MARKET
    timestamp: datetime = None
    order_id: str = ""
    status: str = "pending"  # pending / filled / rejected / cancelled


@dataclass
class Fill:
    """成交"""
    order_id: str
    code: str
    direction: OrderDirection
    price: float
    volume: int
    cost: float
    timestamp: datetime


@dataclass
class Position:
    """持仓"""
    code: str
    name: str = ""
    volume: int = 0          # 持仓数量
    avg_cost: float = 0.0    # 平均成本
    current_price: float = 0.0
    freeze_volume: int = 0   # T+1冻结数量
    pnl: float = 0.0         # 盈亏
    pnl_pct: float = 0.0     # 盈亏百分比


@dataclass
class TradeRecord:
    """成交记录"""
    timestamp: datetime
    code: str
    direction: OrderDirection
    price: float
    volume: int
    cost: float
    pnl: float = 0.0
    balance: float = 0.0


class BacktestEngine:
    """事件驱动回测引擎

    严格遵循 A 股规则：
    - T+1 交易
    - 涨跌停限制
    - 停牌无法交易
    - 手续费/印花税/滑点
    """

    def __init__(self, initial_cash: float = 1_000_000, config: Optional[dict] = None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[dict] = []
        self.current_date: datetime = None
        self.config = config or {}

        # A股规则参数
        self.t_plus_1 = True
        self.limit_up_pct = 0.10      # 主板涨跌幅
        self.limit_down_pct = -0.10
        self.commission_rate = 0.00025  # 万分之2.5
        self.stamp_tax_rate = 0.001   # 千分之一印花税（卖出）
        self.slippage = 0.001         # 滑点 0.1%

        # 统计
        self.total_trades = 0
        self.winning_trades = 0

    def run(self, data: Dict[str, pd.DataFrame], signal_callback: Callable, risk_callback: Optional[Callable] = None) -> dict:
        """运行回测

        Args:
            data: {code: DataFrame} 多股日线数据
            signal_callback: fn(date, positions, cash, data) -> List[Order]
            risk_callback: fn(order, positions, cash) -> bool (return False to reject)

        Returns:
            绩效指标字典
        """
        # 获取所有交易日
        dates = self._get_trading_dates(data)
        self.cash = self.initial_cash

        for date in dates:
            self.current_date = date
            day_data = {code: df.loc[date] for code, df in data.items() if date in df.index}

            # 更新持仓市价
            self._update_positions(day_data)

            # 生成信号
            orders = signal_callback(date, self.positions, self.cash, day_data)

            # 风控校验
            if risk_callback:
                orders = [o for o in orders if risk_callback(o, self.positions, self.cash)]

            # 执行订单
            for order in orders:
                self._execute_order(order, day_data)

            # 记录净值
            self._record_equity()

        return self._calculate_performance()

    def _execute_order(self, order: Order, day_data: Dict[str, dict]):
        """执行订单（A股规则适配）"""
        if order.code not in day_data:
            return

        bar = day_data[order.code]
        price = bar["close"]

        # ---- 涨跌停限制 ----
        if "limit_up" in bar and price >= bar["limit_up"] and order.direction == OrderDirection.BUY:
            return  # 涨停买不进
        if "limit_down" in bar and price <= bar["limit_down"] and order.direction == OrderDirection.SELL:
            return  # 跌停卖不出

        # ---- 滑点 ----
        if order.direction == OrderDirection.BUY:
            exec_price = price * (1 + self.slippage)
        else:
            exec_price = price * (1 - self.slippage)

        # ---- T+1 校验 ----
        pos = self.positions.get(order.code)
        if order.direction == OrderDirection.SELL and pos:
            sellable = pos.volume - pos.freeze_volume
            if sellable <= 0:
                return  # 冻结中无法卖出

        # ---- 计算费用 ----
        volume = order.volume
        amount = exec_price * volume
        commission = amount * self.commission_rate
        stamp_tax = amount * self.stamp_tax_rate if order.direction == OrderDirection.SELL else 0
        total_cost = commission + stamp_tax

        # ---- 执行 ----
        if order.direction == OrderDirection.BUY:
            total_spent = amount + total_cost
            if total_spent > self.cash:
                volume = int((self.cash - total_cost) / exec_price)  # 按可用资金调整
                if volume <= 0:
                    return
                amount = exec_price * volume
                commission = amount * self.commission_rate
                total_spent = amount + commission

            self.cash -= total_spent
            self._update_position(order.code, OrderDirection.BUY, exec_price, volume)

        else:  # SELL
            if pos and pos.volume >= volume:
                sellable = pos.volume - pos.freeze_volume
                if volume > sellable:
                    volume = sellable
                amount = exec_price * volume
                total_received = amount - total_cost
                self.cash += total_received

                pnl = (exec_price - pos.avg_cost) * volume
                self._update_position(order.code, OrderDirection.SELL, exec_price, volume, pnl)

        # 记录成交
        self.trades.append(TradeRecord(
            timestamp=self.current_date,
            code=order.code,
            direction=order.direction,
            price=exec_price,
            volume=volume,
            cost=total_cost,
            pnl=pnl if order.direction == OrderDirection.SELL else 0,
            balance=self.cash,
        ))
        self.total_trades += 1
        if order.direction == OrderDirection.SELL and (pnl if 'pnl' in locals() else 0) > 0:
            self.winning_trades += 1

    def _update_position(self, code: str, direction: OrderDirection, price: float, volume: int, pnl: float = 0):
        """更新持仓"""
        if code not in self.positions:
            self.positions[code] = Position(code=code)

        pos = self.positions[code]
        if direction == OrderDirection.BUY:
            total_cost = pos.avg_cost * pos.volume + price * volume
            pos.volume += volume
            pos.avg_cost = total_cost / pos.volume
            if self.t_plus_1:
                pos.freeze_volume = volume  # 当日买入冻结
        else:
            pos.volume -= volume
            pos.pnl += pnl
            if pos.volume <= 0:
                del self.positions[code]

    def _update_positions(self, day_data: Dict[str, dict]):
        """更新持仓市价和冻结状态"""
        for code, pos in self.positions.items():
            if code in day_data:
                pos.current_price = day_data[code]["close"]
                pos.pnl = (pos.current_price - pos.avg_cost) * pos.volume
                pos.pnl_pct = (pos.current_price / pos.avg_cost - 1) * 100
                # T+1解冻
                pos.freeze_volume = 0

    def _record_equity(self):
        """记录资产净值"""
        market_value = sum(p.volume * p.current_price for p in self.positions.values())
        total = self.cash + market_value
        self.equity_curve.append({
            "date": self.current_date,
            "cash": self.cash,
            "market_value": market_value,
            "total": total,
            "return": (total / self.initial_cash - 1) * 100,
        })

    def _get_trading_dates(self, data: Dict[str, pd.DataFrame]) -> List:
        """获取所有交易日（多股并集）"""
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index if isinstance(df.index, pd.DatetimeIndex) else df["date"])
        return sorted(all_dates)

    def _calculate_performance(self) -> dict:
        """计算核心绩效指标"""
        if not self.equity_curve:
            return {}

        equity = pd.DataFrame(self.equity_curve)
        total_return = (equity["total"].iloc[-1] / self.initial_cash - 1) * 100

        # 最大回撤
        rolling_max = equity["total"].cummax()
        drawdown = (equity["total"] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # 年化收益
        days = len(equity)
        years = days / 252
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        # 夏普比率
        equity["daily_return"] = equity["total"].pct_change()
        sharpe = np.sqrt(252) * equity["daily_return"].mean() / (equity["daily_return"].std() + 1e-8)

        # 胜率
        win_rate = self.winning_trades / self.total_trades * 100 if self.total_trades > 0 else 0

        return {
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": self.total_trades,
            "total_days": days,
            "final_cash": round(self.cash, 2),
        }
