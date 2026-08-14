"""data/factory.py — 因子库封装

内置 100+ 常用量化因子，支持用户自定义因子。
因子分类：趋势、动量、波动、成交量、价值、情绪、形态、复合
"""

from typing import List, Dict, Callable, Optional
import pandas as pd
import numpy as np


# ============================================================================
# 1. 趋势因子 (Trend Factors)
# ============================================================================

def factor_ma(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
    """移动平均线族"""
    for w in windows:
        df[f"MA{w}"] = df["close"].rolling(w).mean()
        df[f"MA{w}_slope"] = df[f"MA{w}"].pct_change(5)  # 均线斜率
    return df


def factor_ema(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """指数移动平均"""
    for w in windows:
        df[f"EMA{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df


def factor_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 指标族"""
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    df["MACD_DIF"] = ema_fast - ema_slow
    df["MACD_DEA"] = df["MACD_DIF"].ewm(span=signal).mean()
    df["MACD_HIST"] = 2 * (df["MACD_DIF"] - df["MACD_DEA"])
    df["MACD_GOLDEN_CROSS"] = ((df["MACD_DIF"] > df["MACD_DEA"]) & (df["MACD_DIF"].shift(1) <= df["MACD_DEA"].shift(1))).astype(int)
    df["MACD_DEAD_CROSS"] = ((df["MACD_DIF"] < df["MACD_DEA"]) & (df["MACD_DIF"].shift(1) >= df["MACD_DEA"].shift(1))).astype(int)
    return df


def factor_price_trend(df: pd.DataFrame) -> pd.DataFrame:
    """价格趋势因子"""
    # 价格位置（当前价在N日高低区间的位置）
    for n in [10, 20, 60]:
        hh = df["high"].rolling(n).max()
        ll = df["low"].rolling(n).min()
        df[f"price_position_{n}"] = (df["close"] - ll) / (hh - ll + 1e-8)
        # 创N日新高/新低
        df[f"new_high_{n}"] = (df["high"] == hh).astype(int)
        df[f"new_low_{n}"] = (df["low"] == ll).astype(int)
    return df


def factor_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX 趋势强度指标"""
    # TODO: 计算 ADX
    pass


# ============================================================================
# 2. 动量因子 (Momentum Factors)
# ============================================================================

def factor_rsi(df: pd.DataFrame, periods: List[int] = [6, 12, 24]) -> pd.DataFrame:
    """RSI 强弱指标族"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for p in periods:
        avg_gain = gain.ewm(span=p).mean()
        avg_loss = loss.ewm(span=p).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df[f"RSI{p}"] = 100 - (100 / (1 + rs))
        df[f"RSI{p}_overbought"] = (df[f"RSI{p}"] > 70).astype(int)
        df[f"RSI{p}_oversold"] = (df[f"RSI{p}"] < 30).astype(int)
    return df


def factor_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """多周期动量"""
    for n in [1, 3, 5, 10, 20, 60]:
        df[f"momentum_{n}d"] = df["close"].pct_change(n)
    return df


def factor_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """KDJ 随机指标"""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-8) * 100
    df["KDJ_K"] = rsv.ewm(span=m1).mean()
    df["KDJ_D"] = df["KDJ_K"].ewm(span=m2).mean()
    df["KDJ_J"] = 3 * df["KDJ_K"] - 2 * df["KDJ_D"]
    return df


def factor_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """威廉指标 W%R"""
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    df["WILLIAMS_R"] = (hh - df["close"]) / (hh - ll + 1e-8) * -100
    return df


def factor_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """CCI 商品通道指数"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    df["CCI"] = (tp - ma) / (0.015 * mad + 1e-8)
    return df


def factor_roc(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """ROC 变动率指标"""
    for p in periods:
        df[f"ROC{p}"] = df["close"].pct_change(p) * 100
    return df


# ============================================================================
# 3. 波动因子 (Volatility Factors)
# ============================================================================

def factor_bollinger(df: pd.DataFrame, period: int = 20, stds: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """布林带族"""
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["BOLL_MID"] = ma
    for s in stds:
        df[f"BOLL_UP_{s}"] = ma + s * std
        df[f"BOLL_DN_{s}"] = ma - s * std
    df["BOLL_WIDTH"] = (df["BOLL_UP_2"] - df["BOLL_DN_2"]) / ma  # 带宽
    df["BOLL_POSITION"] = (df["close"] - df["BOLL_DN_2"]) / (df["BOLL_UP_2"] - df["BOLL_DN_2"] + 1e-8)
    return df


def factor_atr(df: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
    """ATR 平均真实波幅族"""
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for p in periods:
        df[f"ATR{p}"] = tr.rolling(p).mean()
        df[f"ATR{p}_PCT"] = df[f"ATR{p}"] / df["close"]  # 相对波幅
    return df


def factor_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """波动率因子"""
    ret = df["close"].pct_change()
    for n in [5, 10, 20, 60]:
        df[f"volatility_{n}d"] = ret.rolling(n).std()
        df[f"annual_vol_{n}d"] = df[f"volatility_{n}d"] * np.sqrt(252)
    return df


# ============================================================================
# 4. 成交量因子 (Volume Factors)
# ============================================================================

def factor_volume_ma(df: pd.DataFrame) -> pd.DataFrame:
    """均量线族"""
    for w in [5, 10, 20, 60]:
        df[f"VOL_MA{w}"] = df["volume"].rolling(w).mean()
        df[f"VOL_RATIO{w}"] = df["volume"] / (df[f"VOL_MA{w}"] + 1e-8)  # 量比
    return df


def factor_volume_price(df: pd.DataFrame) -> pd.DataFrame:
    """量价关系因子"""
    df["volume_price_trend"] = df["close"].pct_change() * df["volume"].pct_change()  # 量价同向 => 正
    # 成交量标准差
    df["volume_std_20"] = df["volume"].rolling(20).std() / (df["volume"].rolling(20).mean() + 1e-8)
    # 放量/缩量信号
    df["volume_surge"] = (df["volume"] > df["volume"].rolling(20).mean() * 1.5).astype(int)
    df["volume_shrink"] = (df["volume"] < df["volume"].rolling(20).mean() * 0.5).astype(int)
    return df


def factor_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """换手率因子"""
    if "turnover" in df.columns:
        for w in [5, 10, 20]:
            df[f"turnover_ma{w}"] = df["turnover"].rolling(w).mean()
        df["turnover_std"] = df["turnover"].rolling(20).std()
    return df


def factor_money_flow(df: pd.DataFrame) -> pd.DataFrame:
    """资金流因子"""
    if "amount" in df.columns:
        for w in [5, 10, 20]:
            df[f"amount_ma{w}"] = df["amount"].rolling(w).mean()
    return df


# ============================================================================
# 5. 价值因子 (Value Factors - 需接入财务数据)
# ============================================================================

def factor_value(df: pd.DataFrame) -> pd.DataFrame:
    """价值因子（需额外数据源）"""
    # TODO: PE, PB, PS, PC等
    return df


def factor_growth(df: pd.DataFrame) -> pd.DataFrame:
    """成长因子"""
    # TODO: 营收增长率、利润增长率等
    return df


# ============================================================================
# 6. 情绪因子 (Sentiment Factors)
# ============================================================================

def factor_market_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """市场广度因子"""
    # TODO: 涨跌家数、新高新低比例等
    return df


# ============================================================================
# 因子注册表
# ============================================================================

FACTOR_REGISTRY: Dict[str, Callable] = {
    # 趋势因子
    "ma": factor_ma,
    "ema": factor_ema,
    "macd": factor_macd,
    "price_trend": factor_price_trend,
    "adx": factor_adx,
    # 动量因子
    "rsi": factor_rsi,
    "momentum": factor_momentum,
    "kdj": factor_kdj,
    "williams_r": factor_williams_r,
    "cci": factor_cci,
    "roc": factor_roc,
    # 波动因子
    "bollinger": factor_bollinger,
    "atr": factor_atr,
    "volatility": factor_volatility,
    # 成交量因子
    "volume_ma": factor_volume_ma,
    "volume_price": factor_volume_price,
    "turnover": factor_turnover,
    "money_flow": factor_money_flow,
    # 价值因子
    "value": factor_value,
    "growth": factor_growth,
    # 情绪因子
    "market_breadth": factor_market_breadth,
}

# 默认启用的因子组
DEFAULT_FACTOR_GROUPS = [
    "ma", "macd", "rsi", "momentum", "bollinger",
    "atr", "volatility", "volume_ma", "volume_price",
    "price_trend", "turnover", "kdj", "cci", "roc",
]


class FactorEngine:
    """因子计算引擎"""

    def __init__(self, custom_factors: Optional[Dict[str, Callable]] = None):
        self.factors = dict(FACTOR_REGISTRY)
        if custom_factors:
            self.factors.update(custom_factors)

    def compute(self, df: pd.DataFrame, groups: Optional[List[str]] = None) -> pd.DataFrame:
        """计算指定组的因子

        Args:
            df: OHLCV 数据
            groups: 因子组列表，默认使用 DEFAULT_FACTOR_GROUPS

        Returns:
            含因子列的 DataFrame
        """
        if groups is None:
            groups = DEFAULT_FACTOR_GROUPS
        for name in groups:
            if name in self.factors:
                df = self.factors[name](df)
        return df

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有注册因子"""
        for name, fn in self.factors.items():
            df = fn(df)
        return df

    def register_factor(self, name: str, fn: Callable):
        """注册自定义因子"""
        self.factors[name] = fn

    def list_factors(self) -> List[str]:
        """列出所有可用因子"""
        return list(self.factors.keys())
