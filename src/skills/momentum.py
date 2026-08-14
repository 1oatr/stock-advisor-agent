"""skills/momentum.py — 动量分析技能

检测价格动量的强弱、衰竭、切换，预判趋势持续或反转。
"""

from typing import Optional
import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class MomentumAnalysis(BaseSkill):
    """动量分析技能

    - 动量衰竭检测：持续上涨后涨幅收窄 → 趋势可能反转
    - 动量加速检测：涨幅持续扩大 → 趋势加强
    - 多周期动量对比：短期动量 vs 长期动量
    """

    def __init__(self):
        super().__init__(
            name="momentum_analysis",
            description="分析价格动量强弱和衰竭，预判趋势持续或反转",
            category="动量分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 20:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足")

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0

        # ---- 1. 多周期动量 ----
        mom_5d = df["close"].pct_change(5).iloc[-1] if len(df) > 5 else 0
        mom_10d = df["close"].pct_change(10).iloc[-1] if len(df) > 10 else 0
        mom_20d = df["close"].pct_change(20).iloc[-1] if len(df) > 20 else 0

        # 短期动量
        if mom_5d > 0.05:
            patterns.append(f"短期动量向上 (+{mom_5d:.1%})")
            buy_conf += 0.2
        elif mom_5d < -0.05:
            patterns.append(f"短期动量向下 ({mom_5d:.1%})")
            sell_conf += 0.2

        # ---- 2. 动量衰竭检测 ----
        if len(df) > 15:
            recent_returns = df["close"].pct_change().tail(10)

            # 上涨衰竭：前期涨但近期涨幅收窄
            if mom_20d > 0.05:
                recent_3d = df["close"].pct_change(3).iloc[-1]
                prev_3d = df["close"].pct_change(3).iloc[-4] if len(df) > 4 else 0
                if recent_3d < prev_3d and recent_3d < 0:
                    patterns.append("上涨动能衰竭")
                    sell_conf += 0.5

            # 下跌衰竭：前期跌但近期跌幅收窄
            if mom_20d < -0.05:
                recent_3d = df["close"].pct_change(3).iloc[-1]
                prev_3d = df["close"].pct_change(3).iloc[-4] if len(df) > 4 else 0
                if recent_3d > prev_3d and recent_3d > 0:
                    patterns.append("下跌动能衰竭")
                    buy_conf += 0.5

        # ---- 3. 多周期动量一致/背离 ----
        if mom_5d > 0 and mom_10d > 0 and mom_20d > 0:
            patterns.append("多周期动量一致向上")
            buy_conf += 0.35
        elif mom_5d < 0 and mom_10d < 0 and mom_20d < 0:
            patterns.append("多周期动量一致向下")
            sell_conf += 0.35
        elif mom_5d > 0 and mom_20d < 0:
            patterns.append("短期反弹 vs 长期下跌（谨慎）")

        # ---- 4. 均线排列动量 ----
        if "MA5" in df.columns and "MA20" in df.columns and "MA60" in df.columns:
            ma5 = df["MA5"].iloc[-1]
            ma20 = df["MA20"].iloc[-1]
            ma60 = df["MA60"].iloc[-1]
            close = df["close"].iloc[-1]

            if close > ma5 > ma20 > ma60:
                patterns.append("均线多头排列（强势趋势）")
                buy_conf += 0.3
            elif close < ma5 < ma20 < ma60:
                patterns.append("均线空头排列（弱势趋势）")
                sell_conf += 0.3

        # ---- 5. 动量打分 ----
        if buy_conf > sell_conf:
            signal = SIGNAL_BUY
            confidence = min(buy_conf + 0.3, 1.0)
        elif sell_conf > buy_conf:
            signal = SIGNAL_SELL
            confidence = min(sell_conf + 0.3, 1.0)
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"动量分析: {', '.join(patterns)}" if patterns else "动量无明显偏向"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(patterns) / 5.0,
            explanation=explanation,
            patterns_detected=patterns,
        )


class VolumeAnalysis(BaseSkill):
    """量价分析技能

    分析成交量与价格的关系，识别放量突破、缩量调整、量价背离等。
    """

    def __init__(self):
        super().__init__(
            name="volume_analysis",
            description="分析量价关系，识别放量突破、缩量调整、量价背离",
            category="量价分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 20:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足")

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        close = last["close"]
        volume = last["volume"]

        # 均量线
        vol_ma5 = df["volume"].rolling(5).mean().iloc[-1] if len(df) > 5 else volume
        vol_ma20 = df["volume"].rolling(20).mean().iloc[-1] if len(df) > 20 else volume
        vol_ma60 = df["volume"].rolling(60).mean().iloc[-1] if len(df) > 60 else volume

        vol_ratio_5 = volume / (vol_ma5 + 1e-8)
        vol_ratio_20 = volume / (vol_ma20 + 1e-8)

        price_change = df["close"].pct_change().iloc[-1]

        # ---- 1. 放量上涨（健康） ----
        if vol_ratio_20 > 1.5 and price_change > 0.02:
            patterns.append(f"放量上涨 (量比{vol_ratio_20:.1f})")
            buy_conf += 0.55

        # ---- 2. 放量下跌（出货） ----
        if vol_ratio_20 > 1.5 and price_change < -0.02:
            patterns.append(f"放量下跌 (量比{vol_ratio_20:.1f})")
            sell_conf += 0.55

        # ---- 3. 缩量上涨（力度不足） ----
        if vol_ratio_20 < 0.7 and price_change > 0.02:
            patterns.append("缩量上涨（动力不足）")
            sell_conf += 0.35

        # ---- 4. 缩量下跌（惜售/调整） ----
        if vol_ratio_20 < 0.7 and price_change < -0.02:
            patterns.append("缩量下跌（抛压减弱）")
            buy_conf += 0.4

        # ---- 5. 连续放量 ----
        if len(df) > 5:
            recent_vols = df["volume"].tail(5)
            vol_consistency = recent_vols.std() / (recent_vols.mean() + 1e-8)
            if recent_vols.mean() > vol_ma20 * 1.2 and vol_consistency < 0.2:
                patterns.append("持续放量（资金入场）")
                buy_conf += 0.45

        # ---- 6. 天量检测 ----
        if len(df) > 60:
            vol_60_max = df["volume"].tail(60).max()
            if volume >= vol_60_max * 0.95:
                if price_change > 0:
                    patterns.append("天量上涨（可能见顶）")
                    sell_conf += 0.3
                else:
                    patterns.append("天量下跌（恐慌）")
                    sell_conf += 0.4

        # ---- 7. 量价背离 ----
        if len(df) > 10:
            price_trend = df["close"].tail(10).pct_change().sum()
            vol_trend = df["volume"].tail(10).pct_change().sum()
            if price_trend > 0.05 and vol_trend < -0.1:
                patterns.append("量价背离（上涨缩量）")
                sell_conf += 0.45
            elif price_trend < -0.05 and vol_trend > 0.1:
                patterns.append("量价背离（下跌放量）")
                sell_conf += 0.45

        if buy_conf > sell_conf:
            signal = SIGNAL_BUY
            confidence = buy_conf
        elif sell_conf > buy_conf:
            signal = SIGNAL_SELL
            confidence = sell_conf
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"量价分析: {', '.join(patterns)}" if patterns else "量价关系无明显异常"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(patterns) / 4.0,
            explanation=explanation,
            patterns_detected=patterns,
        )
