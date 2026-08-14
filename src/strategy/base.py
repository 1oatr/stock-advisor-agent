"""strategy/base.py — 策略基类

兼容 Backtrader / VectorBT 核心语法的策略开发框架。
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class Signal:
    """交易信号"""
    code: str
    action: str  # buy / sell / hold
    price: float
    volume: int = 0
    position_ratio: float = 0.0
    confidence: float = 0.5
    reason: str = ""
    timestamp: datetime = None


class BaseStrategy:
    """策略基类

    所有策略继承此类，实现 next() 方法即可。

    Usage:
        class MyStrategy(BaseStrategy):
            def next(self, date, data, positions, cash):
                if data["close"] > data["MA5"]:
                    self.buy("000001", ratio=0.1)
    """

    def __init__(self, name: str = "", params: Optional[dict] = None):
        self.name = name
        self.params = params or {}
        self.signals: List[Signal] = []
        self.current_date: datetime = None
        self.current_data: Dict[str, pd.Series] = {}
        self.positions: Dict = {}
        self.cash: float = 0.0

    def next(self, date: datetime, data: Dict[str, pd.Series],
             positions: Dict, cash: float) -> List[Signal]:
        """策略核心逻辑（子类实现）

        Args:
            date: 当前交易日
            data: {field: value} 如 {"close": 10.5, "MA5": 10.2, ...}
            positions: 当前持仓
            cash: 当前现金

        Returns:
            信号列表
        """
        raise NotImplementedError

    def buy(self, code: str, price: float = 0, ratio: float = 0.1,
            reason: str = "") -> Signal:
        """生成买入信号"""
        s = Signal(code=code, action="buy", price=price,
                   position_ratio=ratio, reason=reason, timestamp=self.current_date)
        self.signals.append(s)
        return s

    def sell(self, code: str, price: float = 0, ratio: float = 1.0,
             reason: str = "") -> Signal:
        """生成卖出信号"""
        s = Signal(code=code, action="sell", price=price,
                   position_ratio=ratio, reason=reason, timestamp=self.current_date)
        self.signals.append(s)
        return s

    def hold(self, code: str = "", reason: str = "") -> Signal:
        """生成持有信号"""
        s = Signal(code=code, action="hold", price=0,
                   reason=reason, timestamp=self.current_date)
        return s

    def get_param(self, key: str, default=None):
        """获取策略参数"""
        return self.params.get(key, default)

    def clear_signals(self):
        """清空信号"""
        self.signals = []
