"""strategy/signal.py — 信号生成器

基于策略 + 数据生成交易信号，对接回测引擎和实盘监控。
"""

from typing import Dict, List, Optional, Callable
from .base import BaseStrategy, Signal


class SignalGenerator:
    """信号生成器

    将策略应用于市场数据，生成可执行的交易信号。
    """

    def __init__(self, strategy: BaseStrategy, risk_engine=None):
        self.strategy = strategy
        self.risk_engine = risk_engine

    def generate(self, date, data: Dict[str, dict], positions: Dict, cash: float) -> List[Signal]:
        """生成信号"""
        self.strategy.clear_signals()

        for code, bar in data.items():
            self.strategy.current_date = date
            self.strategy.current_data = bar
            signals = self.strategy.next(date, bar, positions, cash)

        return self.strategy.signals

    def generate_with_risk(self, date, data: Dict[str, dict],
                           positions: Dict, cash: float, risk_ctx) -> List[Signal]:
        """生成信号（通过风控过滤）"""
        signals = self.generate(date, data, positions, cash)

        if self.risk_engine is None:
            return signals

        filtered = []
        for s in signals:
            order_dict = {"code": s.code, "direction": s.action,
                          "price": s.price, "volume": s.volume}
            result = self.risk_engine.check_order(order_dict, risk_ctx)
            if result.approved:
                filtered.append(s)
        return filtered
