"""skills/volatility.py — 波动率分析技能

检测波动率突变、均值回归条件，预判价格波动方向。
"""

from typing import Optional
import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class VolatilityAnalysis(BaseSkill):
    """波动率分析技能

    - 波动率突变：异常高/低波动后的回归
    - ATR 趋势判断：高波动趋势 vs 低波动盘整
    - 布林带收窄/扩张：变盘信号
    """

    def __init__(self):
        super().__init__(
            name="volatility_analysis",
            description="分析波动率变化和均值回归条件，预判价格波动方向",
            category="波动率分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 20:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足")

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0

        # ---- 1. ATR 趋势判断 ----
        if "ATR14" in df.columns:
            atr = df["ATR14"].iloc[-1]
            atr_ma20 = df["ATR14"].rolling(20).mean().iloc[-1] if len(df) > 20 else atr
            atr_ratio = atr / (atr_ma20 + 1e-8)

            if atr_ratio > 1.5:
                patterns.append(f"高波动状态 (ATR比为{atr_ratio:.1f})")
            elif atr_ratio < 0.6:
                patterns.append(f"低波动状态 (ATR比为{atr_ratio:.1f})")
                # 低波动后大概率变盘

        # ---- 2. 布林带收窄/扩张 ----
        if "BOLL_WIDTH" in df.columns:
            current_width = df["BOLL_WIDTH"].iloc[-1]

            if len(df) > 20:
                width_ma = df["BOLL_WIDTH"].rolling(20).mean().iloc[-1]
                width_ratio = current_width / (width_ma + 1e-8)

                if width_ratio < 0.7:
                    # 布林带极度收窄，变盘在即
                    close = df["close"].iloc[-1]
                    boll_mid = df["BOLL_MID"].iloc[-1] if "BOLL_MID" in df.columns else close

                    if close > boll_mid:
                        patterns.append("布林带收窄+价格在中轨之上（向上变盘概率大）")
                        buy_conf += 0.55
                    else:
                        patterns.append("布林带收窄+价格在中轨之下（向下变盘概率大）")
                        sell_conf += 0.55

                elif width_ratio > 1.3:
                    patterns.append("布林带扩张（趋势加速）")
                    if df["close"].iloc[-1] > df["close"].iloc[-5] if len(df) > 5 else False:
                        buy_conf += 0.4
                    else:
                        sell_conf += 0.4

        # ---- 3. 均值回归条件 ----
        if len(df) > 10:
            returns = df["close"].pct_change().dropna()
            recent_ret = returns.tail(5).mean()
            ret_std = returns.tail(20).std() if len(returns) > 20 else returns.std()

            if ret_std > 0:
                z_score = recent_ret / (ret_std + 1e-8)
                if z_score > 2:
                    patterns.append(f"短期涨幅过大 (Z-score={z_score:.1f})，均值回归概率大")
                    sell_conf += 0.5
                elif z_score < -2:
                    patterns.append(f"短期跌幅过大 (Z-score={z_score:.1f})，均值回归概率大")
                    buy_conf += 0.5

        # ---- 4. 波动率突变检测 ----
        if len(df) > 20 and "ATR14" in df.columns:
            atr_values = df["ATR14"].tail(20)
            atr_change = atr_values.pct_change().tail(5)
            atr_acceleration = atr_change.mean()

            if atr_acceleration > 0.3 and atr_values.iloc[-1] > atr_values.mean():
                patterns.append("波动率加速上升（市场情绪激烈）")

        # ---- 综合 ----
        if buy_conf > sell_conf:
            signal = SIGNAL_BUY
            confidence = min(buy_conf + 0.3, 1.0)
        elif sell_conf > buy_conf:
            signal = SIGNAL_SELL
            confidence = min(sell_conf + 0.3, 1.0)
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"波动率分析: {', '.join(patterns)}" if patterns else "波动率状态正常"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(patterns) / 4.0,
            explanation=explanation,
            patterns_detected=patterns,
        )
