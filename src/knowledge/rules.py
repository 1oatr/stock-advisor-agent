"""knowledge/rules.py — 技术分析规则库

定义各种技术分析规则函数，每条规则接收股票数据，返回信号强度和解释。
"""

from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np


# 信号常量
SIGNAL_BUY = "buy"
SIGNAL_SELL = "sell"
SIGNAL_HOLD = "hold"


def rule_ma_trend(df: pd.DataFrame) -> Tuple[str, float, str]:
    """均线趋势规则

    - MA5 > MA20 > MA60 → 上升趋势 (buy)
    - MA5 < MA20 < MA60 → 下降趋势 (sell)
    - 其他 → 震荡 (hold)

    Returns:
        (signal, strength, explanation)
    """
    if df.empty or len(df) < 60:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少60天）"

    # 检查所需列
    required_cols = ["MA5", "MA20", "MA60"]
    if not all(c in df.columns for c in required_cols):
        return SIGNAL_HOLD, 0.0, "缺少均线数据（MA5/MA20/MA60）"

    ma5 = df["MA5"].iloc[-1]
    ma20 = df["MA20"].iloc[-1]
    ma60 = df["MA60"].iloc[-1]
    close = df["close"].iloc[-1]

    # 多头排列：MA5 > MA20 > MA60 且价格在MA5之上
    if close > ma5 > ma20 > ma60:
        # 计算趋势强度：间距越大趋势越强
        spread_5_20 = (ma5 - ma20) / (ma20 + 1e-10)
        spread_20_60 = (ma20 - ma60) / (ma60 + 1e-10)
        strength = min((spread_5_20 + spread_20_60) * 10, 1.0)
        return SIGNAL_BUY, strength, (
            f"均线多头排列（MA5={ma5:.2f} > MA20={ma20:.2f} > MA60={ma60:.2f}），"
            f"价格{close:.2f}在均线之上，上升趋势明确"
        )

    # 空头排列：MA5 < MA20 < MA60
    if close < ma5 < ma20 < ma60:
        spread_5_20 = (ma20 - ma5) / (ma5 + 1e-10)
        spread_20_60 = (ma60 - ma20) / (ma20 + 1e-10)
        strength = min((spread_5_20 + spread_20_60) * 10, 1.0)
        return SIGNAL_SELL, strength, (
            f"均线空头排列（MA5={ma5:.2f} < MA20={ma20:.2f} < MA60={ma60:.2f}），"
            f"价格{close:.2f}在均线之下，下降趋势明确"
        )

    # 部分多头/空头或粘合
    if ma5 > ma20 and ma20 < ma60:
        return SIGNAL_HOLD, 0.3, (
            f"短期向上但长期均线未跟上（MA5={ma5:.2f} > MA20={ma20:.2f}，"
            f"MA60={ma60:.2f}），趋势待确认"
        )
    if ma5 < ma20 and ma20 > ma60:
        return SIGNAL_HOLD, 0.3, (
            f"短期回调但长期趋势未破坏（MA20={ma20:.2f} > MA60={ma60:.2f}），"
            f"持有"
        )

    return SIGNAL_HOLD, 0.2, f"均线粘合盘整 MA5={ma5:.2f} MA20={ma20:.2f} MA60={ma60:.2f}"


def rule_rsi(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> Tuple[str, float, str]:
    """RSI 超买超卖规则

    - RSI < oversold → 超卖买入
    - RSI > overbought → 超买卖出
    - 中间 → 正常持有
    """
    if df.empty or len(df) < 15:
        return SIGNAL_HOLD, 0.0, "数据不足"

    # 找RSI列
    rsi_col = None
    for col in ["RSI14", "RSI", "rsi"]:
        if col in df.columns:
            rsi_col = col
            break
    if rsi_col is None:
        return SIGNAL_HOLD, 0.0, "缺少RSI数据"

    rsi = df[rsi_col].iloc[-1]
    prev_rsi = df[rsi_col].iloc[-2] if len(df) > 1 else rsi

    if rsi < oversold:
        # 超卖越深，买入信号越强
        strength = min((oversold - rsi) / oversold * 1.5, 1.0)
        # 确认RSI拐头
        if rsi > prev_rsi:
            strength = min(strength + 0.2, 1.0)
            return SIGNAL_BUY, strength, (
                f"RSI={rsi:.1f} 进入超卖区（<{oversold}），且RSI拐头向上，"
                f"超跌反弹概率大"
            )
        return SIGNAL_BUY, strength * 0.7, (
            f"RSI={rsi:.1f} 进入超卖区（<{oversold}），但未见明确拐头"
        )

    elif rsi > overbought:
        strength = min((rsi - overbought) / (100 - overbought) * 1.5, 1.0)
        if rsi < prev_rsi:
            strength = min(strength + 0.2, 1.0)
            return SIGNAL_SELL, strength, (
                f"RSI={rsi:.1f} 进入超买区（>{overbought}），且RSI拐头向下，"
                f"回调风险加大"
            )
        return SIGNAL_SELL, strength * 0.7, (
            f"RSI={rsi:.1f} 进入超买区（>{overbought}），但趋势仍在，注意风险"
        )

    # 中间区域，看是否有方向性
    if rsi > 50:
        return SIGNAL_HOLD, 0.4, f"RSI={rsi:.1f} 中性偏强，持有"
    elif rsi < 50:
        return SIGNAL_HOLD, 0.3, f"RSI={rsi:.1f} 中性偏弱，持有"
    return SIGNAL_HOLD, 0.3, f"RSI={rsi:.1f} 中性"


def rule_macd(df: pd.DataFrame) -> Tuple[str, float, str]:
    """MACD 规则

    - DIF 上穿 DEA → 金叉买入
    - DIF 下穿 DEA → 死叉卖出
    - MACD 柱转正/转负 → 辅助信号
    """
    if df.empty or len(df) < 26:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少26天）"

    if "MACD_DIF" not in df.columns or "MACD_DEA" not in df.columns:
        return SIGNAL_HOLD, 0.0, "缺少MACD数据"

    dif = df["MACD_DIF"].iloc[-1]
    dea = df["MACD_DEA"].iloc[-1]
    hist = df["MACD_HIST"].iloc[-1] if "MACD_HIST" in df.columns else dif - dea

    prev_dif = df["MACD_DIF"].iloc[-2] if len(df) > 1 else dif
    prev_dea = df["MACD_DEA"].iloc[-2] if len(df) > 1 else dea

    # 金叉：DIF上穿DEA
    if prev_dif <= prev_dea and dif > dea:
        # 零轴上方金叉更强
        if dif > 0 and dea > 0:
            strength = 0.8
            msg = f"MACD零轴上方金叉（DIF={dif:.3f} ↑ DEA={dea:.3f}），强势买入信号"
        elif dif > 0:
            strength = 0.7
            msg = f"MACD金叉（DIF={dif:.3f} ↑ DEA={dea:.3f}），DIF转正，买入信号"
        else:
            strength = 0.6
            msg = f"MACD零轴下方金叉（DIF={dif:.3f} ↑ DEA={dea:.3f}），反弹信号，注意力度"
        return SIGNAL_BUY, strength, msg

    # 死叉：DIF下穿DEA
    if prev_dif >= prev_dea and dif < dea:
        if dif < 0 and dea < 0:
            strength = 0.8
            msg = f"MACD零轴下方死叉（DIF={dif:.3f} ↓ DEA={dea:.3f}），强烈卖出信号"
        elif dif < 0:
            strength = 0.7
            msg = f"MACD死叉（DIF={dif:.3f} ↓ DEA={dea:.3f}），DIF转负，卖出信号"
        else:
            strength = 0.6
            msg = f"MACD零轴上方死叉（DIF={dif:.3f} ↓ DEA={dea:.3f}），回调信号"
        return SIGNAL_SELL, strength, msg

    # 柱线方向判断
    prev_hist = df["MACD_HIST"].iloc[-2] if len(df) > 1 and "MACD_HIST" in df.columns else hist
    if hist > prev_hist and hist > 0:
        return SIGNAL_BUY, 0.4, f"MACD红柱放大（{prev_hist:.3f}→{hist:.3f}），多头力量增强"
    elif hist < prev_hist and hist < 0:
        return SIGNAL_SELL, 0.4, f"MACD绿柱加深（{prev_hist:.3f}→{hist:.3f}），空头力量增强"

    return SIGNAL_HOLD, 0.3, f"MACD平滑 DIF={dif:.3f} DEA={dea:.3f}"


def rule_bollinger(df: pd.DataFrame) -> Tuple[str, float, str]:
    """布林带规则

    - 价格触及下轨 + 缩口 → 买入
    - 价格触及上轨 + 开口 → 卖出
    - 中轨附近 → 持有
    """
    if df.empty or len(df) < 20:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少20天）"

    required = ["BOLL_MID", "BOLL_UP", "BOLL_DN"]
    if not all(c in df.columns for c in required):
        # 尝试查找带后缀的列名
        for suffix in ["", "_2", "_20_2"]:
            cols = [f"{base}{suffix}" for base in ["BOLL_MID", "BOLL_UP", "BOLL_DN"]]
            if all(c in df.columns for c in cols):
                required = cols
                break
        else:
            return SIGNAL_HOLD, 0.0, "缺少布林带数据"

    mid, up, dn = [df[c].iloc[-1] for c in required]
    close = df["close"].iloc[-1]

    # 布林带宽度
    width = (up - dn) / (mid + 1e-10)
    prev_width = None
    if len(df) > 5:
        prev_up, prev_dn = [df[c].iloc[-5] for c in [required[1], required[2]]]
        prev_width = (prev_up - prev_dn) / (df[required[0]].iloc[-5] + 1e-10)

    # 计算%B
    boll_pos = (close - dn) / (up - dn + 1e-10)

    # 价格跌破下轨
    if close <= dn * 1.01:
        if prev_width is not None and width < prev_width:
            strength = min(0.5 + (dn - close) / (close + 1e-10) * 50, 0.9)
            return SIGNAL_BUY, strength, (
                f"价格{close:.2f}触及布林下轨{dn:.2f}且带宽收窄，"
                f"超卖+缩口=反弹概率高"
            )
        strength = min(0.3 + (dn - close) / (close + 1e-10) * 30, 0.7)
        return SIGNAL_BUY, strength, f"价格{close:.2f}接近布林下轨{dn:.2f}，超卖区域"

    # 价格突破上轨
    if close >= up * 0.99:
        if prev_width is not None and width > prev_width:
            strength = min(0.5 + (close - up) / (close + 1e-10) * 50, 0.9)
            return SIGNAL_SELL, strength, (
                f"价格{close:.2f}突破布林上轨{up:.2f}且带宽扩张，"
                f"超卖+开口=回调风险大"
            )
        strength = min(0.3 + (close - up) / (close + 1e-10) * 30, 0.7)
        return SIGNAL_SELL, strength, f"价格{close:.2f}接近布林上轨{up:.2f}，超买区域"

    # 在通道内
    if boll_pos < 0.3:
        return SIGNAL_BUY, 0.3, f"价格在布林带低位（%B={boll_pos:.0%}），偏多"
    elif boll_pos > 0.7:
        return SIGNAL_SELL, 0.3, f"价格在布林带高位（%B={boll_pos:.0%}），偏空"
    else:
        return SIGNAL_HOLD, 0.3, f"价格在布林带中轨附近（%B={boll_pos:.0%}）"


def rule_volume_price(df: pd.DataFrame) -> Tuple[str, float, str]:
    """量价配合规则

    - 放量上涨 → 确认趋势 (buy)
    - 放量下跌 → 出货信号 (sell)
    - 缩量回调 → 健康调整 (hold)
    - 缩量上涨 → 动能不足 (caution)
    """
    if df.empty or len(df) < 20:
        return SIGNAL_HOLD, 0.0, "数据不足"

    volume = df["volume"]
    close = df["close"]

    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / (vol_ma20 + 1e-10)
    price_change = close.pct_change().iloc[-1]

    # 连续多日量价分析
    if len(df) > 5:
        recent_vol = volume.tail(5)
        recent_price = close.tail(5)
        vol_trend = recent_vol.pct_change().mean()
        price_trend = recent_price.pct_change().mean()

        # 放量上涨
        if vol_ratio > 1.5 and price_change > 0.02:
            if price_trend > 0 and vol_trend > 0:
                return SIGNAL_BUY, 0.7, (
                    f"放量上涨（量比{vol_ratio:.1f}，涨幅{price_change:.1%}），"
                    f"连续放量+价格上涨，资金入场明显"
                )
            return SIGNAL_BUY, 0.55, (
                f"放量上涨（量比{vol_ratio:.1f}，涨幅{price_change:.1%}），多头主动"
            )

        # 放量下跌
        if vol_ratio > 1.5 and price_change < -0.02:
            if price_trend < 0 and vol_trend > 0:
                return SIGNAL_SELL, 0.75, (
                    f"放量下跌（量比{vol_ratio:.1f}，跌幅{price_change:.1%}），"
                    f"连续放量下跌，资金出逃"
                )
            return SIGNAL_SELL, 0.55, (
                f"放量下跌（量比{vol_ratio:.1f}，跌幅{price_change:.1%}），抛压大"
            )

        # 缩量回调（健康）
        if vol_ratio < 0.7 and -0.03 < price_change < 0:
            if price_trend < 0 and vol_trend < 0:
                return SIGNAL_BUY, 0.55, (
                    f"缩量回调（量比{vol_ratio:.1f}，跌幅{price_change:.1%}），"
                    f"缩量下跌说明抛压减弱，调整接近尾声"
                )
            return SIGNAL_HOLD, 0.4, f"缩量回调（量比{vol_ratio:.1f}），持有"

        # 缩量上涨（谨慎）
        if vol_ratio < 0.7 and price_change > 0.02:
            return SIGNAL_SELL, 0.35, (
                f"缩量上涨（量比{vol_ratio:.1f}），上涨无量，动能不足注意回落"
            )

    return SIGNAL_HOLD, 0.2, f"量价正常，量比{vol_ratio:.1f}，涨幅{price_change:.1%}"


def rule_support_resistance(df: pd.DataFrame) -> Tuple[str, float, str]:
    """支撑阻力规则

    识别近期支撑位和阻力位，价格接近支撑位时买入，接近阻力位时卖出。
    """
    if df.empty or len(df) < 30:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少30天）"

    close = df["close"].iloc[-1]

    # 找近期高点/低点作为阻力和支撑
    lookback = min(60, len(df))
    recent = df.tail(lookback)

    # 用滚动窗口找局部极值
    highs = []
    lows = []
    window = 10
    for i in range(window, len(recent) - window):
        if recent["high"].iloc[i] == recent["high"].iloc[i - window:i + window + 1].max():
            highs.append(recent["high"].iloc[i])
        if recent["low"].iloc[i] == recent["low"].iloc[i - window:i + window + 1].min():
            lows.append(recent["low"].iloc[i])

    # 去重和聚合
    def cluster_prices(prices, threshold=0.02):
        if not prices:
            return []
        prices = sorted(set(round(p, 2) for p in prices))
        clusters = []
        current = [prices[0]]
        for p in prices[1:]:
            if abs(p - current[-1]) / (current[-1] + 1e-10) < threshold:
                current.append(p)
            else:
                clusters.append(sum(current) / len(current))
                current = [p]
        clusters.append(sum(current) / len(current))
        return clusters

    resistances = cluster_prices([h for h in highs if h > close])[:3]
    supports = cluster_prices([l for l in lows if l < close])[:3]

    if not supports and not resistances:
        return SIGNAL_HOLD, 0.2, "无法识别明确支撑阻力位"

    nearest_support = max(supports) if supports else None
    nearest_resistance = min(resistances) if resistances else None

    # 计算距当前价格百分比
    signals_found = []

    if nearest_support:
        dist_s = (close - nearest_support) / close * 100
        if dist_s < 1.0:
            signals_found.append(f"距支撑{nearest_support:.2f}仅{dist_s:.1f}%")
            return SIGNAL_BUY, 0.65, f"价格{close:.2f}接近支撑位{nearest_support:.2f}（距{dist_s:.1f}%），支撑有效可低吸"
        elif dist_s < 2.5:
            signals_found.append(f"距支撑{nearest_support:.2f}为{dist_s:.1f}%")
            return SIGNAL_BUY, 0.45, f"价格{close:.2f}接近支撑位{nearest_support:.2f}，可在支撑附近关注"

    if nearest_resistance:
        dist_r = (nearest_resistance - close) / close * 100
        if dist_r < 1.0:
            return SIGNAL_SELL, 0.65, f"价格{close:.2f}接近阻力位{nearest_resistance:.2f}（距{dist_r:.1f}%），阻力明显注意高抛"
        elif dist_r < 2.5:
            return SIGNAL_SELL, 0.45, f"价格{close:.2f}接近阻力位{nearest_resistance:.2f}，可在阻力附近减持"

    # 在支撑和阻力之间
    if nearest_support and nearest_resistance:
        range_pct = (close - nearest_support) / (nearest_resistance - nearest_support) * 100
        if range_pct < 30:
            return SIGNAL_BUY, 0.35, f"价格在区间低位（{range_pct:.0f}%），支撑{nearest_support:.2f}~阻力{nearest_resistance:.2f}"
        elif range_pct > 70:
            return SIGNAL_SELL, 0.35, f"价格在区间高位（{range_pct:.0f}%），支撑{nearest_support:.2f}~阻力{nearest_resistance:.2f}"
        else:
            return SIGNAL_HOLD, 0.2, f"价格在区间中部（{range_pct:.0f}%），支撑{nearest_support:.2f}~阻力{nearest_resistance:.2f}"

    return SIGNAL_HOLD, 0.2, f"支撑{nearest_support or '无'} / 阻力{nearest_resistance or '无'}"


def rule_stop_loss(df: pd.DataFrame, entry_price: float = 0.0,
                   threshold: float = 0.05) -> Tuple[str, float, str]:
    """止损规则（否决级风控）

    亏损超过阈值 → 强制卖出，直接否决所有买入信号。
    无持仓时返回 hold(0)，不影响正常评分。

    Args:
        df: 股票数据
        entry_price: 持仓成本价（0 表示无持仓）
        threshold: 止损阈值
    """
    if entry_price <= 0 or df.empty:
        return SIGNAL_HOLD, 0.0, "无持仓或无数据，止损规则不适用"

    current_price = float(df["close"].iloc[-1])
    loss_pct = (current_price - entry_price) / entry_price

    if loss_pct <= -threshold:
        urgency = min(abs(loss_pct) / threshold, 2.0)
        strength = min(urgency * 0.6, 1.0)
        return SIGNAL_SELL, strength, (
            f"触发止损！亏损{loss_pct:.1%}（阈值{threshold:.0%}），"
            f"当前{current_price:.2f} / 成本{entry_price:.2f}"
        )

    if loss_pct <= -threshold * 0.7:
        return SIGNAL_SELL, 0.3, (
            f"接近止损线（亏损{loss_pct:.1%}，阈值{threshold:.0%}），"
            f"注意风险"
        )

    return SIGNAL_HOLD, 0.2, f"距止损线{abs(loss_pct/threshold-1)*100:.0f}%"


def rule_kdj(df: pd.DataFrame, oversold: float = 20, overbought: float = 80) -> Tuple[str, float, str]:
    """KDJ 随机指标规则

    A 股最常用的短线指标，比 RSI 更灵敏。
    - K 上穿 D（金叉）+ K < 30 → 低位金叉买入
    - K 下穿 D（死叉）+ K > 70 → 高位死叉卖出
    - J < 0 → 极度超卖
    - J > 100 → 极度超买
    - J 值拐头确认 → 增强信号

    Returns:
        (signal, strength, explanation)
    """
    if df.empty or len(df) < 9:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少9天）"

    required = ["KDJ_K", "KDJ_D", "KDJ_J"]
    if not all(c in df.columns for c in required):
        return SIGNAL_HOLD, 0.0, "缺少KDJ数据（KDJ_K/KDJ_D/KDJ_J）"

    k = df["KDJ_K"].iloc[-1]
    d = df["KDJ_D"].iloc[-1]
    j = df["KDJ_J"].iloc[-1]

    prev_k = df["KDJ_K"].iloc[-2] if len(df) > 1 else k
    prev_d = df["KDJ_D"].iloc[-2] if len(df) > 1 else d
    prev_j = df["KDJ_J"].iloc[-2] if len(df) > 1 else j

    # ── KDJ 金叉：K 上穿 D ──
    if prev_k <= prev_d and k > d:
        if k < oversold + 10:  # K < 30 → 低位金叉
            strength = 0.80
            msg = (f"KDJ低位金叉（K={k:.1f}↑D={d:.1f}，J={j:.1f}），"
                   f"超卖区反转，强烈买入信号")
        elif k < 50:
            strength = 0.65
            msg = (f"KDJ中位金叉（K={k:.1f}↑D={d:.1f}，J={j:.1f}），"
                   f"买入信号")
        else:
            strength = 0.50
            msg = (f"KDJ高位金叉（K={k:.1f}↑D={d:.1f}，J={j:.1f}），"
                   f"高位金叉注意追高风险")
        return SIGNAL_BUY, strength, msg

    # ── KDJ 死叉：K 下穿 D ──
    if prev_k >= prev_d and k < d:
        if k > overbought - 10:  # K > 70 → 高位死叉
            strength = 0.80
            msg = (f"KDJ高位死叉（K={k:.1f}↓D={d:.1f}，J={j:.1f}），"
                   f"超买区反转，强烈卖出信号")
        elif k > 50:
            strength = 0.65
            msg = (f"KDJ中位死叉（K={k:.1f}↓D={d:.1f}，J={j:.1f}），"
                   f"卖出信号")
        else:
            strength = 0.50
            msg = (f"KDJ低位死叉（K={k:.1f}↓D={d:.1f}，J={j:.1f}），"
                   f"低位死叉可能为洗盘，注意观察")
        return SIGNAL_SELL, strength, msg

    # ── J 值极端信号（无交叉时的辅助判断）──
    if j < 0:
        if j > prev_j:  # J 从负值拐头 → 超卖反弹
            strength = min(0.35 + abs(j) / 40, 0.75)
            return SIGNAL_BUY, strength, (
                f"KDJ_J={j:.1f}<0 极度超卖且拐头向上，"
                f"反弹概率大"
            )
        return SIGNAL_BUY, 0.40, f"KDJ_J={j:.1f}<0 极度超卖区，等待拐头确认"

    if j > 100:
        if j < prev_j:  # J 从高位回落 → 超买回调
            strength = min(0.35 + (j - 100) / 40, 0.75)
            return SIGNAL_SELL, strength, (
                f"KDJ_J={j:.1f}>100 极度超买且拐头向下，"
                f"回调风险大"
            )
        return SIGNAL_SELL, 0.40, f"KDJ_J={j:.1f}>100 极度超买区，等待拐头确认"

    # ── K 值区域判断 ──
    if k < oversold:
        return SIGNAL_BUY, 0.35, (
            f"KDJ低位钝化 K={k:.1f} D={d:.1f} J={j:.1f}，"
            f"关注金叉信号"
        )
    if k > overbought:
        return SIGNAL_SELL, 0.35, (
            f"KDJ高位钝化 K={k:.1f} D={d:.1f} J={j:.1f}，"
            f"关注死叉信号"
        )

    return SIGNAL_HOLD, 0.25, f"KDJ中性 K={k:.1f} D={d:.1f} J={j:.1f}"


def rule_adx(df: pd.DataFrame) -> Tuple[str, float, str]:
    """ADX 趋势强度 + DI 方向规则

    双重逻辑：
    1. ADX 判断趋势强度（过滤震荡市无效信号）
    2. +DI / -DI 判断趋势方向

    - ADX > 25 且 +DI > -DI → 上升趋势确认
    - ADX > 25 且 -DI > +DI → 下降趋势确认
    - ADX < 20 → 无趋势/震荡，降低所有信号权重（此规则输出 hold）

    Returns:
        (signal, strength, explanation)
    """
    if df.empty or len(df) < 30:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少30天）"

    required = ["ADX", "ADX_PLUS_DI", "ADX_MINUS_DI"]
    if not all(c in df.columns for c in required):
        return SIGNAL_HOLD, 0.0, "缺少ADX数据（ADX/ADX_PLUS_DI/ADX_MINUS_DI）"

    adx = df["ADX"].iloc[-1]
    plus_di = df["ADX_PLUS_DI"].iloc[-1]
    minus_di = df["ADX_MINUS_DI"].iloc[-1]

    # 近5日ADX趋势（上升=趋势增强，下降=趋势减弱）
    if len(df) >= 5:
        adx_5d_ago = df["ADX"].iloc[-5]
        adx_slope = (adx - adx_5d_ago) / (adx_5d_ago + 1e-10)
    else:
        adx_slope = 0.0

    # ── 无趋势/震荡 ──
    if adx < 20:
        return SIGNAL_HOLD, 0.15, (
            f"ADX={adx:.1f}<20 无明显趋势，市场处于震荡状态，"
            f"趋势类信号可信度降低"
        )

    # ── 弱趋势（ADX 20-25）──
    if adx < 25:
        if plus_di > minus_di:
            di_diff = (plus_di - minus_di) / (plus_di + 1e-10)
            strength = min(0.25 + di_diff * 2, 0.50)
            return SIGNAL_BUY, strength, (
                f"ADX={adx:.1f} 趋势偏弱但+DI({plus_di:.1f})> -DI({minus_di:.1f})，"
                f"多头略占优，建议配合其他信号"
            )
        elif minus_di > plus_di:
            di_diff = (minus_di - plus_di) / (minus_di + 1e-10)
            strength = min(0.25 + di_diff * 2, 0.50)
            return SIGNAL_SELL, strength, (
                f"ADX={adx:.1f} 趋势偏弱但-DI({minus_di:.1f})> +DI({plus_di:.1f})，"
                f"空头略占优，建议配合其他信号"
            )
        return SIGNAL_HOLD, 0.20, f"ADX={adx:.1f} 趋势偏弱，DI方向不明确"

    # ── 强趋势（ADX > 40）──
    if adx > 40:
        if plus_di > minus_di:
            di_ratio = plus_di / (minus_di + 1e-10)
            strength = min(0.60 + (adx - 40) / 30 + (di_ratio - 1) * 0.5, 0.95)
            trend_note = "增强" if adx_slope > 0.05 else ("减弱" if adx_slope < -0.05 else "持续")
            return SIGNAL_BUY, strength, (
                f"ADX={adx:.1f}>40 强上升趋势（趋势{trend_note}），"
                f"+DI({plus_di:.1f})>>-DI({minus_di:.1f})，多头主导"
            )
        elif minus_di > plus_di:
            di_ratio = minus_di / (plus_di + 1e-10)
            strength = min(0.60 + (adx - 40) / 30 + (di_ratio - 1) * 0.5, 0.95)
            trend_note = "增强" if adx_slope > 0.05 else ("减弱" if adx_slope < -0.05 else "持续")
            return SIGNAL_SELL, strength, (
                f"ADX={adx:.1f}>40 强下降趋势（趋势{trend_note}），"
                f"-DI({minus_di:.1f})>>+DI({plus_di:.1f})，空头主导"
            )

    # ── 正常趋势（ADX 25-40）──
    if plus_di > minus_di:
        di_ratio = plus_di / (minus_di + 1e-10)
        strength = min(0.40 + di_ratio * 0.20, 0.70)
        return SIGNAL_BUY, strength, (
            f"ADX={adx:.1f} 上升趋势明确，+DI({plus_di:.1f})> -DI({minus_di:.1f})"
        )
    elif minus_di > plus_di:
        di_ratio = minus_di / (plus_di + 1e-10)
        strength = min(0.40 + di_ratio * 0.20, 0.70)
        return SIGNAL_SELL, strength, (
            f"ADX={adx:.1f} 下降趋势明确，-DI({minus_di:.1f})> +DI({plus_di:.1f})"
        )

    return SIGNAL_HOLD, 0.25, f"ADX={adx:.1f} 方向不明 +DI={plus_di:.1f} -DI={minus_di:.1f}"


def rule_limit_gap(df: pd.DataFrame, code: str = "") -> Tuple[str, float, str]:
    """涨跌停 + 跳空缺口检测规则（A 股特色）

    - 涨停/连板 → 强势持有（但追高风险提示）
    - 跌停/连板 → 强烈卖出
    - 向上跳空缺口 → 突破确认
    - 向下跳空缺口 → 破位确认

    涨跌停幅度根据板块自动判断：
    - 300xxx/301xxx/688xxx → 20%
    - 其他 → 10%
    """
    if df.empty or len(df) < 5:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少5天）"

    close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    prev_high = df["high"].iloc[-2]
    prev_low = df["low"].iloc[-2]

    # 判断涨跌停幅度
    if code and (code.startswith("300") or code.startswith("301") or code.startswith("688")):
        limit_pct = 0.20
    else:
        limit_pct = 0.10

    change_pct = (close - prev_close) / prev_close
    is_limit_up = change_pct >= limit_pct * 0.99
    is_limit_down = change_pct <= -limit_pct * 0.99

    # 连续涨跌停计数
    consecutive_limits = 0
    if len(df) >= 5:
        for i in range(1, min(5, len(df))):
            past_close = df["close"].iloc[-(i + 1)]
            past_prev = df["close"].iloc[-(i + 2)] if len(df) > i + 1 else past_close
            past_change = (past_close - past_prev) / (past_prev + 1e-10)
            if is_limit_up and past_change >= limit_pct * 0.98:
                consecutive_limits += 1
            elif is_limit_down and past_change <= -limit_pct * 0.98:
                consecutive_limits += 1
            else:
                break

    if is_limit_up:
        if consecutive_limits >= 2:
            return SIGNAL_BUY, 0.85, (
                f"连续{consecutive_limits + 1}个涨停板（+{change_pct:.1%}），"
                f"极度强势但追高风险极大，开板即应减持"
            )
        return SIGNAL_BUY, 0.70, (
            f"涨停板（+{change_pct:.1%}），封板强势，"
            f"短线不宜追涨，等待开板回调再评估"
        )

    if is_limit_down:
        if consecutive_limits >= 2:
            return SIGNAL_SELL, 0.95, (
                f"连续{consecutive_limits + 1}个跌停板（{change_pct:.1%}），"
                f"恐慌抛售，不要接飞刀，等待缩量企稳"
            )
        return SIGNAL_SELL, 0.75, (
            f"跌停板（{change_pct:.1%}），恐慌信号，如有持仓尽快止损"
        )

    # ── 跳空缺口 ──
    gap_up = low > prev_high
    gap_down = high < prev_low

    if gap_up:
        gap_size = (low - prev_high) / prev_high * 100
        return SIGNAL_BUY, 0.55, (
            f"向上跳空缺口（{gap_size:.2f}%），突破缺口，"
            f"若3日内不回补则确认突破有效"
        )

    if gap_down:
        gap_size = (prev_low - high) / prev_low * 100
        return SIGNAL_SELL, 0.60, (
            f"向下跳空缺口（{gap_size:.2f}%），破位信号，大概率继续下行"
        )

    # 接近涨停/跌停（未封板）
    if change_pct > limit_pct * 0.7:
        return SIGNAL_BUY, 0.40, f"接近涨停（+{change_pct:.1%}），但未封板，注意回落"
    if change_pct < -limit_pct * 0.7:
        return SIGNAL_SELL, 0.45, f"接近跌停（{change_pct:.1%}），卖压极重"

    return SIGNAL_HOLD, 0.15, "无涨跌停或明显缺口"


def rule_volume_anomaly(df: pd.DataFrame) -> Tuple[str, float, str]:
    """成交量异常检测规则

    检测 6 种深度成交量异常（比基础 rule_volume_price 更深层）：
    1. 天量天价 → 头部信号
    2. 地量地价 → 底部信号
    3. 放量滞涨 → 出货信号
    4. 缩量止跌 → 企稳信号
    5. 量价背离（价涨量缩多日）→ 动能衰竭
    6. 极端放量超过均量3倍 → 重大变盘
    """
    if df.empty or len(df) < 20:
        return SIGNAL_HOLD, 0.0, "数据不足（需至少20天）"

    volume = df["volume"]
    close = df["close"]

    vol_latest = volume.iloc[-1]
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    vol_ma60 = volume.rolling(60).mean().iloc[-1] if len(df) >= 60 else vol_ma20
    vol_max60 = volume.rolling(60).max().iloc[-1] if len(df) >= 60 else volume.max()
    vol_min60 = volume.rolling(60).min().iloc[-1] if len(df) >= 60 else volume.min()

    price_change_1d = close.pct_change().iloc[-1]
    price_change_5d = (close.iloc[-1] / (close.iloc[-5] + 1e-10) - 1) if len(df) >= 5 else 0
    price_change_10d = (close.iloc[-1] / (close.iloc[-10] + 1e-10) - 1) if len(df) >= 10 else 0

    # 量趋势
    vol_5d_avg = volume.tail(5).mean()
    vol_prev_5d_avg = volume.tail(10).head(5).mean() if len(df) >= 10 else vol_5d_avg
    vol_trend = vol_5d_avg / (vol_prev_5d_avg + 1e-10) - 1

    # 60日价格位置
    if len(df) >= 60:
        price_high = close.rolling(60).max().iloc[-1]
        price_low = close.rolling(60).min().iloc[-1]
        price_pos = (close.iloc[-1] - price_low) / (price_high - price_low + 1e-10)
    else:
        price_pos = 0.5

    signals_found = []

    # ── 1. 天量天价 ──
    if vol_latest >= vol_max60 * 0.95 and price_pos > 0.70:
        if price_change_1d < 0.02:
            signals_found.append(("sell", 0.60, (
                f"天量天价：60日最高量 + 60日高位{price_pos:.0%}，"
                f"主力出货嫌疑，注意回避"
            )))
        else:
            signals_found.append(("sell", 0.45, (
                f"高位放量：60日最高量 + 60日高位{price_pos:.0%}，关注是否滞涨"
            )))

    # ── 2. 地量地价 ──
    if vol_latest <= vol_min60 * 1.1 and price_pos < 0.30:
        signals_found.append(("buy", 0.55, (
            f"地量地价：60日最低量 + 60日低位{price_pos:.0%}，"
            f"抛压枯竭，底部区域信号"
        )))

    # ── 3. 放量滞涨 ──
    if vol_latest > vol_ma20 * 1.5 and abs(price_change_1d) < 0.01 and price_change_5d < 0.03:
        signals_found.append(("sell", 0.55, (
            f"放量滞涨：量比{vol_latest / (vol_ma20 + 1e-10):.1f}但涨幅仅{price_change_1d:.1%}，"
            f"主力对倒出货特征明显"
        )))

    # ── 4. 缩量止跌 ──
    if vol_trend < -0.20 and -0.03 < price_change_5d < 0:
        signals_found.append(("buy", 0.50, (
            f"缩量止跌：近5日量缩{abs(vol_trend):.0%}，跌幅收窄至{price_change_5d:.1%}，"
            f"有望企稳反弹"
        )))

    # ── 5. 量价背离 ──
    if price_change_10d > 0.05 and vol_trend < -0.15:
        signals_found.append(("sell", 0.50, (
            f"量价背离：10日涨{price_change_10d:.1%}但近5日量缩{abs(vol_trend):.0%}，"
            f"上涨动能衰竭，回调风险增加"
        )))

    # ── 6. 极端放量（>3倍）──
    if vol_latest > vol_ma20 * 3.0:
        if price_change_1d > 0.03:
            signals_found.append(("buy", 0.50, (
                f"极端放量上涨：量比{vol_latest / (vol_ma20 + 1e-10):.1f}，"
                f"可能为机构大额买入，但需警惕一日游"
            )))
        else:
            signals_found.append(("sell", 0.55, (
                f"极端放量：量比{vol_latest / (vol_ma20 + 1e-10):.1f}，"
                f"异常放量需密切关注方向选择"
            )))

    if not signals_found:
        return SIGNAL_HOLD, 0.15, f"量价正常，量比{vol_latest / (vol_ma20 + 1e-10):.1f}"

    buy_signals = [(s, e) for d, s, e in signals_found if d == "buy"]
    sell_signals = [(s, e) for d, s, e in signals_found if d == "sell"]

    if buy_signals and (not sell_signals or max(s for s, _ in buy_signals) >= max(s for s, _ in sell_signals)):
        best = max(buy_signals, key=lambda x: x[0])
        return SIGNAL_BUY, best[0], best[1]
    elif sell_signals:
        best = max(sell_signals, key=lambda x: x[0])
        return SIGNAL_SELL, best[0], best[1]

    return SIGNAL_HOLD, 0.20, f"量价异动但方向不明确，量比{vol_latest / (vol_ma20 + 1e-10):.1f}"


# ============================================================================
# 规则注册表
# ============================================================================

RULE_REGISTRY = [
    # === 趋势跟踪类 (权重合计 ~0.38) ===
    {"name": "ma_trend", "fn": rule_ma_trend, "weight": 0.16, "description": "均线趋势"},
    {"name": "macd", "fn": rule_macd, "weight": 0.13, "description": "MACD金叉死叉"},
    {"name": "adx", "fn": rule_adx, "weight": 0.09, "description": "ADX趋势强度+方向"},

    # === 震荡指标类 (权重合计 ~0.20) ===
    {"name": "kdj", "fn": rule_kdj, "weight": 0.11, "description": "KDJ超买超卖+金叉死叉"},
    {"name": "rsi", "fn": rule_rsi, "weight": 0.09, "description": "RSI超买超卖"},

    # === 通道/波动类 (权重合计 ~0.10) ===
    {"name": "bollinger", "fn": rule_bollinger, "weight": 0.10, "description": "布林带通道"},

    # === 量价类 (权重合计 ~0.16) ===
    {"name": "volume_price", "fn": rule_volume_price, "weight": 0.09, "description": "量价配合"},
    {"name": "volume_anomaly", "fn": rule_volume_anomaly, "weight": 0.07, "description": "成交量异常（天量/地量/滞涨/背离）"},

    # === 价格结构类 (权重合计 ~0.14) ===
    {"name": "support_resistance", "fn": rule_support_resistance, "weight": 0.08, "description": "支撑阻力"},
    {"name": "limit_gap", "fn": rule_limit_gap, "weight": 0.06, "description": "涨跌停+跳空缺口"},
]

# 否决级规则：触发后直接覆盖综合信号（不参与加权投票）
VETO_RULES = [
    {"name": "stop_loss", "fn": rule_stop_loss, "description": "止损（否决买入）"},
]
