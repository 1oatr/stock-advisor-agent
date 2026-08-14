"""scanner/criteria.py — 热门股票筛选条件

定义各种筛选条件函数，可组合使用。
"""

from typing import Dict, Union, Optional
import pandas as pd
import numpy as np


def volume_surge(df: pd.DataFrame, ratio: float = 1.5, lookback: int = 20) -> Union[bool, float]:
    """量比突增：当日成交量 > 日均成交量 * ratio

    Returns:
        bool / 放量倍数（False时返回0）
    """
    if df.empty or len(df) < lookback:
        return False

    vol = df["volume"]
    vol_ma = vol.tail(lookback).mean()
    current_vol = vol.iloc[-1]

    if vol_ma == 0:
        return False

    vol_ratio = current_vol / vol_ma
    return vol_ratio if vol_ratio >= ratio else False


def price_momentum(df: pd.DataFrame, days: int = 5, threshold: float = 0.05) -> Union[bool, float]:
    """价格动量：过去 days 天涨幅超过 threshold

    Returns:
        bool / 涨幅值（False时返回0）
    """
    if df.empty or len(df) < days:
        return False

    ret = df["close"].pct_change(days).iloc[-1]
    return ret if abs(ret) >= threshold else False


def ma_bullish(df: pd.DataFrame) -> Union[bool, float]:
    """均线多头排列：MA5 > MA20 > MA60 且收盘价均在之上

    Returns:
        bool / 多头强度评分 (0~1)
    """
    required = ["MA5", "MA20", "MA60"]
    if df.empty or len(df) < 60 or not all(c in df.columns for c in required):
        return False

    ma5 = df["MA5"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    ma60 = df["MA60"].iloc[-1]
    close = df["close"].iloc[-1]

    if close > ma5 > ma20 > ma60:
        # 计算排列紧密度，越分散评分越高
        spread = (ma5 - ma20) / ma20 + (ma20 - ma60) / ma60
        return min(spread * 5, 1.0)
    return False


def ma_bearish(df: pd.DataFrame) -> Union[bool, float]:
    """均线空头排列：MA5 < MA20 < MA60"""
    required = ["MA5", "MA20", "MA60"]
    if df.empty or len(df) < 60 or not all(c in df.columns for c in required):
        return False

    ma5 = df["MA5"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    ma60 = df["MA60"].iloc[-1]
    close = df["close"].iloc[-1]

    if close < ma5 < ma20 < ma60:
        spread = (ma20 - ma5) / ma5 + (ma60 - ma20) / ma20
        return min(spread * 5, 1.0)
    return False


def breakout_high(df: pd.DataFrame, lookback: int = 20) -> Union[bool, float]:
    """突破近期高点：收盘价创 lookback 日新高

    Returns:
        bool / 突破幅度 (%)
    """
    if df.empty or len(df) < lookback:
        return False

    recent_high = df["high"].iloc[-lookback:-1].max()
    close = df["close"].iloc[-1]

    if close > recent_high:
        return (close / recent_high - 1) * 100
    return False


def breakout_low(df: pd.DataFrame, lookback: int = 20) -> Union[bool, float]:
    """跌破近期低点：收盘价创 lookback 日新低"""
    if df.empty or len(df) < lookback:
        return False

    recent_low = df["low"].iloc[-lookback:-1].min()
    close = df["close"].iloc[-1]

    if close < recent_low:
        return (recent_low / close - 1) * 100
    return False


def rsi_signal(df: pd.DataFrame, lower: float = 30, upper: float = 70) -> str:
    """RSI 信号：返回 \"oversold\" / \"overbought\" / \"normal\""""
    if df.empty or len(df) < 15:
        return "normal"

    rsi_col = None
    for col in ["RSI14", "RSI", "rsi"]:
        if col in df.columns:
            rsi_col = col
            break
    if rsi_col is None:
        return "normal"

    rsi = df[rsi_col].iloc[-1]
    if rsi < lower:
        return "oversold"
    elif rsi > upper:
        return "overbought"
    return "normal"


def macd_signal(df: pd.DataFrame) -> str:
    """MACD 信号：返回 \"golden_cross\" / \"dead_cross\" / \"none\""""
    if df.empty or len(df) < 26:
        return "none"

    if "MACD_DIF" not in df.columns or "MACD_DEA" not in df.columns:
        return "none"

    dif = df["MACD_DIF"].iloc[-1]
    dea = df["MACD_DEA"].iloc[-1]
    prev_dif = df["MACD_DIF"].iloc[-2] if len(df) > 1 else dif
    prev_dea = df["MACD_DEA"].iloc[-2] if len(df) > 1 else dea

    if prev_dif <= prev_dea and dif > dea:
        return "golden_cross"
    elif prev_dif >= prev_dea and dif < dea:
        return "dead_cross"
    return "none"


# 筛选条件注册表
CRITERIA_REGISTRY = {
    "volume_surge": volume_surge,
    "price_momentum": price_momentum,
    "ma_bullish": ma_bullish,
    "ma_bearish": ma_bearish,
    "breakout_high": breakout_high,
    "breakout_low": breakout_low,
    "rsi_signal": rsi_signal,
    "macd_signal": macd_signal,
}


def evaluate_all(df: pd.DataFrame) -> Dict[str, any]:
    """运行所有筛选条件，返回结果字典"""
    results = {}
    for name, fn in CRITERIA_REGISTRY.items():
        try:
            results[name] = fn(df)
        except Exception as e:
            results[name] = f"error: {e}"
    return results
