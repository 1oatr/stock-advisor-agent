"""skills/volatility_regime.py — 波动率区间判断技能

不只是看布林带宽度，而是将市场状态分为三种波动率区间：
- 低波（压缩）：ATR% 持续低位 → 布林收窄 → 即将变盘
- 正常：正常波动范围，适合常规交易
- 高波（恐慌）：ATR% 飙升 → 恐慌或亢奋，风险极高

这个技能的特殊之处：不判断方向，而是判断"是否适合交易"。
可作为其他技能信号的质量过滤器。
"""

import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class VolatilityRegimeSkill(BaseSkill):
    """波动率区间判断

    三种区间：
    - 低波压缩 → 布林带收窄，变盘在即 → 方向其他技能定，但此时突破信号可靠性更高
    - 正常区间 → 正常交易
    - 高波恐慌 → 不适合建仓，已有持仓考虑减持
    """

    # 可配置阈值
    ATR_LOW_PCT = 1.5       # ATR% < 1.5% 为低波（100元股票日内波动<1.5元）
    ATR_HIGH_PCT = 3.5      # ATR% > 3.5% 为高波
    LOW_VOL_MIN_DAYS = 5    # 低波需持续至少5天
    BB_WIDTH_LOW = 5.0      # 布林带宽 < 5% 为压缩信号

    def __init__(self):
        super().__init__(
            name="volatility_regime",
            description="判断波动率区间（压缩/正常/恐慌），决定是否适合交易",
            category="风控分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 20:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.0, explanation="数据不足",
            )

        patterns = []
        regime = "normal"
        buy_conf = 0.0
        sell_conf = 0.0

        # ---- 1. 获取 ATR% ----
        atr_pct_col = "ATR_PCT" if "ATR_PCT" in df.columns else None
        if atr_pct_col is None:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.0, strength=0.0,
                explanation="缺少 ATR_PCT 指标，请先运行 add_all_indicators()",
            )

        current_atr_pct = float(df[atr_pct_col].iloc[-1])
        atr_series = df[atr_pct_col].tail(20).astype(float)

        # ---- 2. 布林带宽 ----
        boll_width = None
        if "BOLL_UP" in df.columns and "BOLL_DN" in df.columns:
            boll_mid = df["BOLL_MID"].iloc[-1]
            if boll_mid and boll_mid > 0:
                boll_width = (df["BOLL_UP"].iloc[-1] - df["BOLL_DN"].iloc[-1]) / boll_mid * 100

        # ---- 3. 判断区间 ----
        atr_mean_10d = atr_series.tail(10).mean()
        atr_min_10d = atr_series.tail(10).min()
        atr_max_20d = atr_series.max()
        atr_20d_ago = atr_series.iloc[0] if len(atr_series) >= 20 else atr_mean_10d
        atr_change = (current_atr_pct / (atr_20d_ago + 1e-8) - 1) * 100

        # 低波压缩判断
        all_recent_atr = df[atr_pct_col].tail(self.LOW_VOL_MIN_DAYS).astype(float)
        is_low_sustained = all(all_recent_atr < self.ATR_LOW_PCT)

        # 布林带宽也收窄
        boll_squeezing = boll_width is not None and boll_width < self.BB_WIDTH_LOW

        if is_low_sustained:
            regime = "low_vol_squeeze"
            patterns.append(f"低波压缩已持续{self.LOW_VOL_MIN_DAYS}天 (ATR%={current_atr_pct:.1f}%)")

            if boll_squeezing:
                patterns.append(f"布林带宽收窄至{boll_width:.1f}%，变盘在即")

            # 低波时不做方向判断，但标记为"关注"
            # 结合价格趋势给微弱方向
            price_5d = df["close"].pct_change(5).iloc[-1] * 100 if len(df) > 5 else 0
            if price_5d > 3:
                patterns.append("压缩区间内价格偏强，关注向上突破")
                buy_conf += 0.30
            elif price_5d < -3:
                patterns.append("压缩区间内价格偏弱，关注向下突破")
                sell_conf += 0.30
            else:
                patterns.append("方向不明，等待突破方向确认")

            # 低波本身是一个"准备交易"的信号，不是方向信号
            confidence = 0.55

        # 高波恐慌判断
        elif current_atr_pct > self.ATR_HIGH_PCT:
            regime = "high_vol_panic"

            # 区分恐慌与亢奋
            price_5d = df["close"].pct_change(5).iloc[-1] * 100 if len(df) > 5 else 0

            if price_5d < -5:
                patterns.append(f"高波恐慌 (ATR%={current_atr_pct:.1f}%, 5日跌{price_5d:.1f}%)")
                patterns.append("恐慌抛售中，不宜抄底，等待波动率回归")
                sell_conf += 0.40
            elif price_5d > 5:
                patterns.append(f"高波亢奋 (ATR%={current_atr_pct:.1f}%, 5日涨{price_5d:.1f}%)")
                patterns.append("放量急涨，追高风险大，可减持锁定利润")
                sell_conf += 0.30
            else:
                patterns.append(f"波动率异常放大 (ATR%={current_atr_pct:.1f}%)")
                patterns.append("波动率异常但方向不明确，建议持有")

            # ATR 飙升程度
            if atr_change > 50:
                patterns.append(f"ATR较20日前骤增{atr_change:.0f}%，风险剧增")
                sell_conf += 0.20

            confidence = 0.65

        # 正常区间
        else:
            regime = "normal"
            # ATR 变化趋势
            if atr_change > 20:
                patterns.append(f"波动率上升中 (ATR%={current_atr_pct:.1f}%, +{atr_change:.0f}%)")
            elif atr_change < -20:
                patterns.append(f"波动率下降中 (ATR%={current_atr_pct:.1f}%, {atr_change:.0f}%)")

            if boll_width and boll_width > 15:
                patterns.append(f"布林带宽偏宽({boll_width:.1f}%)，波动空间大")

            # 正常区间不产生买卖信号，中立
            buy_conf = 0.0
            sell_conf = 0.0
            confidence = 0.50

        # ---- 4. 综合 ----
        if buy_conf > sell_conf and buy_conf > 0.2:
            signal = SIGNAL_BUY
            conf = min(buy_conf, 0.8)
        elif sell_conf > buy_conf and sell_conf > 0.2:
            signal = SIGNAL_SELL
            conf = min(sell_conf, 0.8)
        else:
            signal = SIGNAL_HOLD
            conf = confidence

        if not patterns:
            patterns.append("波动率处于正常区间，常规交易模式")

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=conf,
            strength=max(buy_conf, sell_conf, 0.3),
            explanation="; ".join(patterns),
            patterns_detected=patterns,
            metadata={
                "regime": regime,
                "atr_pct": round(current_atr_pct, 2),
                "boll_width": round(boll_width, 2) if boll_width else None,
                "atr_change_pct": round(atr_change, 1),
            },
        )
