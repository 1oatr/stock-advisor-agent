"""skills/divergence.py — 背离检测技能

检测价格与指标之间的背离，预测趋势反转。
"""

from typing import Optional
import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class RSIDivergence(BaseSkill):
    """RSI 背离检测

    - 顶背离：价格创新高，RSI未创新高 → 看跌反转
    - 底背离：价格创新低，RSI未创新低 → 看涨反转
    """

    def __init__(self):
        super().__init__(
            name="rsi_divergence",
            description="检测价格与RSI的顶/底背离，预判趋势反转",
            category="动量分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 30 or "RSI14" not in df.columns:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足或缺少RSI指标")

        rsi_col = "RSI14"
        lookback = 20

        # 取最近 lookback 天的数据
        recent = df.tail(lookback + 5)

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0

        # ---- 顶背离检测 ----
        # 价格创近期新高，但RSI未创新高
        price_high_idx = recent["high"].idxmax()
        price_high = recent["high"].max()
        price_high_pos = recent.index.get_loc(price_high_idx)

        rsi_window = recent.iloc[:price_high_pos + 1]
        if len(rsi_window) > 3:
            rsi_high = rsi_window[rsi_col].max()
            rsi_high_idx = rsi_window[rsi_col].idxmax()

            # 价格新高发生在RSI新高之后 → 顶背离
            if (recent[rsi_col].iloc[-1] < rsi_high - 5
                    and recent["close"].iloc[-1] >= price_high * 0.98):
                patterns.append("RSI顶背离（价格新高，RSI未确认）")
                sell_conf = 0.75

        # ---- 底背离检测 ----
        # 价格创近期新低，但RSI未创新低
        price_low_idx = recent["low"].idxmin()
        price_low_pos = recent.index.get_loc(price_low_idx)

        rsi_window = recent.iloc[:price_low_pos + 1]
        if len(rsi_window) > 3:
            rsi_low = rsi_window[rsi_col].min()
            rsi_low_idx = rsi_window[rsi_col].idxmin()

            if (recent[rsi_col].iloc[-1] > rsi_low + 5
                    and recent["close"].iloc[-1] <= price_low * 1.02):
                patterns.append("RSI底背离（价格新低，RSI已企稳）")
                buy_conf = 0.75

        # ---- RSI 常规超买超卖 ----
        current_rsi = recent[rsi_col].iloc[-1]
        if current_rsi < 25:
            patterns.append(f"RSI超卖 ({current_rsi:.0f})")
            buy_conf = max(buy_conf, 0.6)
        elif current_rsi > 75:
            patterns.append(f"RSI超买 ({current_rsi:.0f})")
            sell_conf = max(sell_conf, 0.6)

        if buy_conf > sell_conf:
            signal = SIGNAL_BUY
            confidence = buy_conf
        elif sell_conf > buy_conf:
            signal = SIGNAL_SELL
            confidence = sell_conf
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"检测到 {', '.join(patterns)}" if patterns else "未检测到明显背离"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(patterns) / 3.0,
            explanation=explanation,
            patterns_detected=patterns,
        )


class MACDDivergence(BaseSkill):
    """MACD 背离检测

    - 顶背离：价格新高，MACD柱/DIF未新高
    - 底背离：价格新低，MACD柱/DIF未新低
    """

    def __init__(self):
        super().__init__(
            name="macd_divergence",
            description="检测价格与MACD的顶/底背离，预判趋势转折",
            category="动量分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 30 or "MACD_DIF" not in df.columns:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足或缺少MACD指标")

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0
        recent = df.tail(30)

        # 顶背离：价格高点 > 前高点，MACD高点 < 前高点
        price_peaks = self._find_peaks(recent["high"].values, window=5)
        macd_peaks = self._find_peaks(recent["MACD_DIF"].values, window=5)

        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            last_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            # 找到对应时间附近的MACD峰值
            macd_at_price_peak = recent["MACD_DIF"].iloc[last_price_peak]
            macd_at_prev_peak = recent["MACD_DIF"].iloc[prev_price_peak]

            if (recent["high"].iloc[last_price_peak] > recent["high"].iloc[prev_price_peak]
                    and macd_at_price_peak < macd_at_prev_peak):
                patterns.append("MACD顶背离")
                sell_conf = 0.7

        # 底背离
        price_valleys = self._find_valleys(recent["low"].values, window=5)
        macd_valleys = self._find_valleys(recent["MACD_DIF"].values, window=5)

        if len(price_valleys) >= 2 and len(macd_valleys) >= 2:
            last_valley = price_valleys[-1]
            prev_valley = price_valleys[-2]
            macd_at_valley = recent["MACD_DIF"].iloc[last_valley]
            macd_at_prev = recent["MACD_DIF"].iloc[prev_valley]

            if (recent["low"].iloc[last_valley] < recent["low"].iloc[prev_valley]
                    and macd_at_valley > macd_at_prev):
                patterns.append("MACD底背离")
                buy_conf = 0.7

        # MACD金叉/死叉
        if "MACD_GOLDEN_CROSS" in df.columns and df["MACD_GOLDEN_CROSS"].iloc[-1] == 1:
            patterns.append("MACD金叉")
            buy_conf = max(buy_conf, 0.6)
        if "MACD_DEAD_CROSS" in df.columns and df["MACD_DEAD_CROSS"].iloc[-1] == 1:
            patterns.append("MACD死叉")
            sell_conf = max(sell_conf, 0.6)

        if buy_conf > sell_conf:
            signal = SIGNAL_BUY
            confidence = buy_conf
        elif sell_conf > buy_conf:
            signal = SIGNAL_SELL
            confidence = sell_conf
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"检测到 {', '.join(patterns)}" if patterns else "未检测到明显MACD背离信号"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(patterns) / 3.0,
            explanation=explanation,
            patterns_detected=patterns,
        )

    def _find_peaks(self, arr, window=5):
        """查找数组中的峰值索引"""
        peaks = []
        for i in range(window, len(arr) - window):
            if arr[i] == max(arr[i - window:i + window + 1]):
                peaks.append(i)
        return peaks

    def _find_valleys(self, arr, window=5):
        """查找数组中的谷值索引"""
        valleys = []
        for i in range(window, len(arr) - window):
            if arr[i] == min(arr[i - window:i + window + 1]):
                valleys.append(i)
        return valleys
