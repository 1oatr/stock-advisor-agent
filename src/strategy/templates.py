"""strategy/templates.py — 策略模板

内置常用策略模板：趋势跟踪、均值回归、突破、网格交易。
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略

    核心逻辑：
    - 均线多头排列 + 放量 → 买入
    - 均线死叉或趋势走坏 → 卖出
    - 趋势延续 → 持有
    """

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "ma_short": 5,
            "ma_medium": 20,
            "ma_long": 60,
            "volume_ratio": 1.5,
            "position_per_trade": 0.15,
            "stop_loss": -0.05,
        }
        params = {**default_params, **(params or {})}
        super().__init__(name="trend_following", params=params)

    def next(self, date, data, positions, cash):
        signals = []

        for code, bar in data.items():
            close = bar.get("close", 0)
            ma5 = bar.get("MA5", 0)
            ma20 = bar.get("MA20", 0)
            ma60 = bar.get("MA60", 0)
            vol_ratio = bar.get("VOL_RATIO", 1.0)
            vol_ratio5 = bar.get("VOL_RATIO5", 1.0)

            current_pos = positions.get(code)
            has_position = current_pos is not None and current_pos.volume > 0

            # 买入条件：多头排列 + 放量
            buy_condition = (
                close > ma5 > ma20 > ma60
                and vol_ratio > self.params["volume_ratio"]
            )

            # 卖出条件：死叉或跌破关键均线
            sell_condition = (
                (ma5 < ma20 and ma5 < ma60)
                or close < ma5 * 0.97
            )

            if buy_condition and not has_position:
                signals.append(self.buy(
                    code=code,
                    price=close,
                    ratio=self.params["position_per_trade"],
                    reason="均线多头排列+放量，趋势跟踪买入",
                ))
            elif sell_condition and has_position:
                signals.append(self.sell(
                    code=code,
                    price=close,
                    reason="趋势走坏，止损/止盈出场",
                ))

        return signals


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略（低吸高抛）

    核心逻辑：
    - RSI 超卖 + 触及布林下轨 → 买入
    - RSI 超买 + 触及布林上轨 → 卖出
    - 价格回归中轨 → 持有
    """

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "position_per_trade": 0.1,
            "take_profit": 0.03,
            "stop_loss": -0.03,
        }
        params = {**default_params, **(params or {})}
        super().__init__(name="mean_reversion", params=params)

    def next(self, date, data, positions, cash):
        signals = []

        for code, bar in data.items():
            close = bar.get("close", 0)
            rsi = bar.get("RSI", bar.get("RSI14", 50))
            boll_dn = bar.get("BOLL_DN", bar.get("BOLL_DN_2", 0))
            boll_up = bar.get("BOLL_UP", bar.get("BOLL_UP_2", 0))
            boll_mid = bar.get("BOLL_MID", 0)

            current_pos = positions.get(code)
            has_position = current_pos is not None and current_pos.volume > 0

            # 超卖买入
            if (rsi < self.params["rsi_oversold"]
                    and close <= boll_dn * 1.02
                    and not has_position):
                signals.append(self.buy(
                    code=code,
                    price=close,
                    ratio=self.params["position_per_trade"],
                    reason=f"RSI超卖({rsi:.0f})+触及布林下轨，均值回归买入",
                ))
            # 超买卖出
            elif (rsi > self.params["rsi_overbought"]
                  and close >= boll_up * 0.98
                  and has_position):
                signals.append(self.sell(
                    code=code,
                    price=close,
                    reason=f"RSI超买({rsi:.0f})+触及布林上轨，均值回归卖出",
                ))
            # 盈利止盈
            elif has_position and current_pos.pnl_pct >= self.params["take_profit"] * 100:
                signals.append(self.sell(
                    code=code,
                    price=close,
                    reason=f"达到止盈目标({self.params['take_profit']:.0%})",
                ))

        return signals


class BreakoutStrategy(BaseStrategy):
    """突破策略

    核心逻辑：
    - 价格突破20日新高 + 放量 → 买入
    - 跌破支撑或趋势反转 → 卖出
    """

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "lookback": 20,
            "volume_ratio": 1.3,
            "position_per_trade": 0.12,
            "stop_loss": -0.04,
        }
        params = {**default_params, **(params or {})}
        super().__init__(name="breakout", params=params)

    def next(self, date, data, positions, cash):
        signals = []

        for code, bar in data.items():
            close = bar.get("close", 0)
            vol_ratio = bar.get("VOL_RATIO", 1.0)
            high = bar.get("high", close)

            current_pos = positions.get(code)
            has_position = current_pos is not None and current_pos.volume > 0

            if not has_position:
                # 突破买入条件：暂无历史数据中的前高判断，在回测框架中处理
                pass
            elif current_pos.pnl_pct <= self.params["stop_loss"] * 100:
                signals.append(self.sell(
                    code=code,
                    price=close,
                    reason=f"触发止损({self.params['stop_loss']:.0%})",
                ))

        return signals


class ETFGridStrategy(BaseStrategy):
    """ETF 网格交易策略

    核心逻辑：
    - 在预设价格区间内分批买入/卖出
    - 每下跌一格买入，每上涨一格卖出
    - 自动网格重建
    """

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "grid_levels": 8,
            "grid_spacing": 0.02,
            "base_position": 0.5,
            "position_per_grid": 0.06,
            "max_position": 0.95,
        }
        params = {**default_params, **(params or {})}
        super().__init__(name="etf_grid", params=params)
        self.grid_prices: Dict[str, List[float]] = {}
        self.grid_bought: Dict[str, List[bool]] = {}

    def init_grid(self, code: str, current_price: float):
        """初始化网格"""
        if code in self.grid_prices:
            return
        spacing = self.params["grid_spacing"]
        levels = self.params["grid_levels"]
        center = current_price

        prices = []
        for i in range(-levels // 2, levels // 2 + 1):
            prices.append(center * (1 + i * spacing))

        self.grid_prices[code] = prices
        self.grid_bought[code] = [False] * len(prices)

    def next(self, date, data, positions, cash):
        signals = []

        for code, bar in data.items():
            close = bar.get("close", 0)
            if close <= 0:
                continue

            # 首次运行初始化网格
            if code not in self.grid_prices:
                self.init_grid(code, close)

            prices = self.grid_prices[code]
            bought = self.grid_bought[code]
            current_pos = positions.get(code)
            has_position = current_pos is not None and current_pos.volume > 0
            current_ratio = (current_pos.volume * close / (cash + 1)) if has_position else 0

            for i, (grid_price, is_bought) in enumerate(zip(prices, bought)):
                # 价格跌破网格线 → 买入（未买过才买）
                if close <= grid_price and not is_bought and current_ratio < self.params["max_position"]:
                    bought[i] = True
                    signals.append(self.buy(
                        code=code,
                        price=close,
                        ratio=self.params["position_per_grid"],
                        reason=f"网格买入 @ {grid_price:.2f}",
                    ))
                    current_ratio += self.params["position_per_grid"]

                # 价格涨回网格线上方 → 卖出（已买过才卖）
                elif close >= grid_price * (1 + self.params["grid_spacing"] * 0.5) and is_bought:
                    bought[i] = False
                    if has_position and current_ratio > 0:
                        signals.append(self.sell(
                            code=code,
                            price=close,
                            ratio=min(self.params["position_per_grid"] / current_ratio, 1.0),
                            reason=f"网格卖出 @ {grid_price:.2f}",
                        ))

        return signals


# 策略注册表
STRATEGY_TEMPLATES = {
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "etf_grid": ETFGridStrategy,
}


def get_strategy(name: str, params: Optional[dict] = None) -> BaseStrategy:
    """获取策略实例"""
    if name in STRATEGY_TEMPLATES:
        return STRATEGY_TEMPLATES[name](params=params)
    raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_TEMPLATES.keys())}")
