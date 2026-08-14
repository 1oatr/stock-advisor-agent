"""skills/breakout.py — 突破分析技能

检测价格突破关键技术位（均线、趋势线、通道、前高/前低），预判趋势加速或反转。
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class BreakoutAnalysis(BaseSkill):
    """突破分析技能

    - 均线突破：价格突破关键均线
    - 通道突破：布林带/通道突破
    - 前高/前低突破
    - 趋势线突破
    """

    def __init__(self):
        super().__init__(
            name="breakout_analysis",
            description="检测突破均线、通道、前高前低等关键位，预判趋势加速",
            category="趋势分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 30:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足")

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        close = last["close"]

        # ---- 1. 均线突破 ----
        for ma_name, ma_period, weight in [("MA5", 5, 0.1), ("MA20", 20, 0.2), ("MA60", 60, 0.3)]:
            if ma_name in df.columns:
                ma_val = df[ma_name].iloc[-1]
                prev_ma = df[ma_name].iloc[-2] if len(df) > 2 else ma_val

                # 上穿
                if prev is not None and prev["close"] <= prev_ma and close > ma_val:
                    patterns.append(f"上穿{ma_name} ({ma_period}日)")
                    buy_conf += weight
                # 下穿
                if prev is not None and prev["close"] >= prev_ma and close < ma_val:
                    patterns.append(f"下穿{ma_name} ({ma_period}日)")
                    sell_conf += weight

        # ---- 2. 布林带突破 ----
        if "BOLL_UP_2" in df.columns and "BOLL_DN_2" in df.columns:
            boll_up = df["BOLL_UP_2"].iloc[-1]
            boll_dn = df["BOLL_DN_2"].iloc[-1]
            boll_mid = df["BOLL_MID"].iloc[-1] if "BOLL_MID" in df.columns else None

            if close > boll_up:
                patterns.append("突破布林带上轨（强势）")
                buy_conf += 0.3
            elif close < boll_dn:
                patterns.append("跌破布林带下轨（弱势）")
                sell_conf += 0.3
            elif boll_mid and prev is not None:
                # 中轨支撑/压力
                if prev["close"] <= boll_mid and close > boll_mid:
                    patterns.append("站上布林带中轨")
                    buy_conf += 0.2
                elif prev["close"] >= boll_mid and close < boll_mid:
                    patterns.append("跌破布林带中轨")
                    sell_conf += 0.2

        # ---- 3. 前高/前低突破 ----
        lookback = 20
        if len(df) > lookback:
            recent_high = df["high"].iloc[-lookback:-1].max()
            recent_low = df["low"].iloc[-lookback:-1].min()

            if close > recent_high:
                patterns.append(f"突破{lookback}日新高")
                buy_conf += 0.25
            elif close < recent_low:
                patterns.append(f"跌破{lookback}日新低")
                sell_conf += 0.25

        # ---- 4. 价格位置分析 ----
        if "BOLL_WIDTH" in df.columns:
            band_width = df["BOLL_WIDTH"].iloc[-1]
            if band_width < band_width * 0.8:  # 布林带收窄
                # 即将变盘
                if buy_conf > sell_conf:
                    patterns.append("布林带收窄+向上（变盘前兆）")
                else:
                    patterns.append("布林带收窄+向下（变盘前兆）")

        # ---- 综合判断 ----
        if buy_conf > sell_conf and buy_conf > 0.3:
            signal = SIGNAL_BUY
            confidence = min(buy_conf + 0.3, 1.0)
        elif sell_conf > buy_conf and sell_conf > 0.3:
            signal = SIGNAL_SELL
            confidence = min(sell_conf + 0.3, 1.0)
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"检测到 {len(patterns)} 个突破信号: {', '.join(patterns)}" if patterns else "无明显突破信号"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(patterns) / 5.0,
            explanation=explanation,
            patterns_detected=patterns,
        )


class SupportResistance(BaseSkill):
    """支撑阻力分析技能

    自动识别关键支撑/阻力位，判断当前价格在区间中的位置。
    """

    def __init__(self):
        super().__init__(
            name="support_resistance",
            description="识别关键支撑位和阻力位，预判价格目标",
            category="趋势分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 30:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足")

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0

        close = df["close"].iloc[-1]

        # ---- 获取关键价位 ----
        levels = self._find_key_levels(df)
        support_levels = levels.get("supports", [])
        resistance_levels = levels.get("resistances", [])

        if not support_levels and not resistance_levels:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.3, explanation="无法识别关键价位")

        # ---- 最近支撑/阻力 ----
        nearest_support = max([s for s in support_levels if s < close], default=None)
        nearest_resistance = min([r for r in resistance_levels if r > close], default=None)

        if nearest_support:
            dist_to_support = (close - nearest_support) / close * 100
            if dist_to_support < 2:
                patterns.append(f"接近支撑位 {nearest_support:.2f}（距当前 {dist_to_support:.1f}%）")
                if dist_to_support < 1:
                    buy_conf += 0.6  # 接近支撑买入
                else:
                    buy_conf += 0.4

        if nearest_resistance:
            dist_to_resistance = (nearest_resistance - close) / close * 100
            if dist_to_resistance < 2:
                patterns.append(f"接近阻力位 {nearest_resistance:.2f}（距当前 {dist_to_resistance:.1f}%）")
                if dist_to_resistance < 1:
                    sell_conf += 0.6
                else:
                    sell_conf += 0.4

        # ---- 价格在区间中的位置 ----
        if nearest_support and nearest_resistance:
            range_high = nearest_resistance
            range_low = nearest_support
            position = (close - range_low) / (range_high - range_low)

            if position < 0.2:
                patterns.append("价格处于区间低位")
                buy_conf = max(buy_conf, 0.5)
            elif position > 0.8:
                patterns.append("价格处于区间高位")
                sell_conf = max(sell_conf, 0.5)
            else:
                patterns.append(f"价格处于区间中部 (位置: {position:.0%})")

        if buy_conf > sell_conf:
            signal = SIGNAL_BUY
            confidence = buy_conf
        elif sell_conf > buy_conf:
            signal = SIGNAL_SELL
            confidence = sell_conf
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"支撑 {nearest_support:.2f} → 当前 {close:.2f} → 阻力 {nearest_resistance:.2f}" if nearest_support and nearest_resistance else "支撑/阻力位识别完成"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=0.5,
            explanation=explanation,
            patterns_detected=patterns,
        )

    def _find_key_levels(self, df: pd.DataFrame, lookback: int = 60) -> dict:
        """自动识别关键支撑阻力位

        使用价格聚类 + 成交量加权识别重要价位。
        """
        recent = df.tail(lookback)
        prices = recent[["high", "low", "close"]].values.flatten()
        volumes = recent["volume"].values

        # 简单方法：取前高、前低、整数关口
        highs = recent["high"].nlargest(5).tolist()
        lows = recent["low"].nsmallest(5).tolist()

        # 取最近的几个高/低点
        recent_highs = recent["high"].rolling(10).max().dropna().unique()
        recent_lows = recent["low"].rolling(10).min().dropna().unique()

        resistances = sorted(set(
            round(h, 2) for h in highs + recent_highs.tolist()
            if h > df["close"].iloc[-1]
        ), reverse=True)[:3]

        supports = sorted(set(
            round(l, 2) for l in lows + recent_lows.tolist()
            if l < df["close"].iloc[-1]
        ))[:3]

        return {"supports": supports, "resistances": resistances}
