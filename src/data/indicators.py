"""data/indicators.py — 技术指标计算

基于 pandas-ta + 自定义计算，提供常用技术分析指标。
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """批量添加所有常用技术指标

    Args:
        df: 必须含 open, high, low, close, volume

    Returns:
        添加所有指标列后的 DataFrame
    """
    df = df.copy()

    # 趋势
    df = add_ma(df)
    df = add_macd(df)

    # 动量
    df = add_rsi(df)
    df = add_kdj(df)
    df = add_cci(df)

    # 波动
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_volatility(df)

    # 成交量
    df = add_volume_indicators(df)

    # 价格位置
    df = add_price_position(df)

    # 日涨跌幅（RL 训练需要）
    df["price_change"] = df["close"].pct_change(1) * 100

    return df


# ============================================================================
# 趋势指标
# ============================================================================

def add_ma(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60, 120]) -> pd.DataFrame:
    """移动平均线族 + 均线斜率"""
    for w in windows:
        df[f"MA{w}"] = df["close"].rolling(window=w, min_periods=w // 2).mean()
        if len(df) > w + 5:
            df[f"MA{w}_slope"] = (df[f"MA{w}"] - df[f"MA{w}"].shift(5)) / (df[f"MA{w}"].shift(5) + 1e-10) * 100
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 指标（DIF, DEA, MACD柱, 金叉死叉）"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD_DIF"] = ema_fast - ema_slow
    df["MACD_DEA"] = df["MACD_DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD_HIST"] = 2 * (df["MACD_DIF"] - df["MACD_DEA"])
    # 金叉/死叉
    cross = df["MACD_DIF"] - df["MACD_DEA"]
    df["MACD_GOLDEN_CROSS"] = ((cross > 0) & (cross.shift(1) <= 0)).astype(int)
    df["MACD_DEAD_CROSS"] = ((cross < 0) & (cross.shift(1) >= 0)).astype(int)
    return df


def add_ema(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """指数移动平均"""
    for w in windows:
        df[f"EMA{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


# ============================================================================
# 动量指标
# ============================================================================

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI 相对强弱指标"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_OVERBOUGHT"] = (df["RSI"] > 70).astype(int)
    df["RSI_OVERSOLD"] = (df["RSI"] < 30).astype(int)
    return df


def add_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ 随机指标"""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
    df["KDJ_K"] = rsv.ewm(span=m1, adjust=False).mean()
    df["KDJ_D"] = df["KDJ_K"].ewm(span=m2, adjust=False).mean()
    df["KDJ_J"] = 3 * df["KDJ_K"] - 2 * df["KDJ_D"]
    df["KDJ_OVERBOUGHT"] = (df["KDJ_K"] > 80).astype(int)
    df["KDJ_OVERSOLD"] = (df["KDJ_K"] < 20).astype(int)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """CCI 商品通道指数"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - ma) / (0.015 * mad + 1e-10)
    df["CCI_OVERBOUGHT"] = (df["CCI"] > 100).astype(int)
    df["CCI_OVERSOLD"] = (df["CCI"] < -100).astype(int)
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """多周期动量"""
    for n in [1, 3, 5, 10, 20, 60]:
        df[f"MOM_{n}d"] = df["close"].pct_change(n) * 100
    return df


# ============================================================================
# 波动指标
# ============================================================================

def add_bollinger(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
    """布林带（上轨、中轨、下轨、带宽、%B）"""
    ma = df["close"].rolling(period).mean()
    std_val = df["close"].rolling(period).std()
    df["BOLL_MID"] = ma
    df["BOLL_UP"] = ma + std * std_val
    df["BOLL_DN"] = ma - std * std_val
    df["BOLL_WIDTH"] = (df["BOLL_UP"] - df["BOLL_DN"]) / (ma + 1e-10)
    df["BOLL_POSITION"] = (df["close"] - df["BOLL_DN"]) / (df["BOLL_UP"] - df["BOLL_DN"] + 1e-10)
    df["BOLL_HIT_UPPER"] = (df["close"] >= df["BOLL_UP"] * 0.99).astype(int)
    df["BOLL_HIT_LOWER"] = (df["close"] <= df["BOLL_DN"] * 1.01).astype(int)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR 平均真实波幅"""
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(period).mean()
    df["ATR_PCT"] = df["ATR"] / (df["close"] + 1e-10) * 100  # 相对波幅
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX 平均趋向指数 + DI 线

    ADX 衡量趋势强度（不判断方向）：
    - ADX > 25 → 趋势明显
    - ADX > 40 → 强趋势
    - ADX < 20 → 无趋势/震荡

    +DI / -DI 判断趋势方向：
    - +DI > -DI → 多头主导
    - -DI > +DI → 空头主导
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder's smoothing (EMA with alpha = 1/period)
    atr_smooth = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / (atr_smooth + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / (atr_smooth + 1e-10)

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    df["ADX"] = adx
    df["ADX_PLUS_DI"] = plus_di
    df["ADX_MINUS_DI"] = minus_di
    df["ADX_TRENDING"] = (adx > 25).astype(int)
    df["ADX_STRONG"] = (adx > 40).astype(int)

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """波动率（历史波动率）"""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for n in [5, 10, 20, 60]:
        df[f"VOLATILITY_{n}d"] = log_ret.rolling(n).std() * np.sqrt(252) * 100
    return df


# ============================================================================
# 成交量指标
# ============================================================================

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """成交量指标（均量线、量比、量价关系）"""
    for w in [5, 10, 20, 60]:
        df[f"VOL_MA{w}"] = df["volume"].rolling(w).mean()
        df[f"VOL_RATIO{w}"] = df["volume"] / (df[f"VOL_MA{w}"] + 1e-10)

    # 量比（当日/20日均量）
    df["VOL_RATIO"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
    df["VOL_SURGE"] = (df["VOL_RATIO"] > 1.5).astype(int)
    df["VOL_SHRINK"] = (df["VOL_RATIO"] < 0.5).astype(int)

    # 金额均线
    if "amount" in df.columns:
        for w in [5, 20]:
            df[f"AMOUNT_MA{w}"] = df["amount"].rolling(w).mean()

    return df


# ============================================================================
# 价格位置
# ============================================================================

def add_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """价格在N日高低区间的相对位置"""
    for n in [10, 20, 60]:
        hh = df["high"].rolling(n).max()
        ll = df["low"].rolling(n).min()
        df[f"PRICE_POS_{n}d"] = (df["close"] - ll) / (hh - ll + 1e-10)
        df[f"NEW_HIGH_{n}d"] = (df["high"] == hh).astype(int)
        df[f"NEW_LOW_{n}d"] = (df["low"] == ll).astype(int)

    return df
