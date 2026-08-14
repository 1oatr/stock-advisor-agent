"""skills/candlestick.py — K线形态识别技能

识别常见K线组合形态，预测短期走势反转或延续。
"""

from typing import Optional
import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class CandlestickPatterns(BaseSkill):
    """K线形态识别

    检测单根/多根K线组合形态：
    - 反转形态: 锤子线、上吊线、吞没形态、十字星、启明星、黄昏星
    - 延续形态: 前进三兵、三只乌鸦
    - 单根形态: 大阳线/大阴线、长上影/长下影
    """

    def __init__(self):
        super().__init__(
            name="candlestick_patterns",
            description="识别K线组合形态（锤子线、吞没、十字星等），预测短期反转",
            category="形态识别",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 5:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="数据不足")

        patterns_detected = []
        buy_signals = []
        sell_signals = []

        # 获取最近几根K线
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        prev2 = df.iloc[-3] if len(df) > 2 else None
        prev3 = df.iloc[-4] if len(df) > 3 else None

        body = abs(last["close"] - last["open"])
        upper_shadow = last["high"] - max(last["close"], last["open"])
        lower_shadow = min(last["close"], last["open"]) - last["low"]
        total_range = last["high"] - last["low"]
        is_green = last["close"] > last["open"]

        if total_range == 0:
            return SkillResult(skill_name=self.name, signal=SIGNAL_HOLD,
                               confidence=0.0, explanation="价格无波动")

        shadow_ratio_body = body / total_range if total_range > 0 else 0

        # ---- 1. 锤子线 / 上吊线 ----
        # 下影线长（>实体2倍）、上影线短、实体在顶部
        if (lower_shadow >= 2 * body and upper_shadow < body * 0.3
                and shadow_ratio_body < 0.4):
            if prev is not None and prev["close"] < prev["open"]:  # 下跌后出现
                patterns_detected.append("锤子线（看涨反转）")
                buy_signals.append(0.7)
            else:
                patterns_detected.append("上吊线（看跌反转）")
                sell_signals.append(0.6)

        # ---- 2. 吞没形态 ----
        if prev is not None:
            prev_body = abs(prev["close"] - prev["open"])
            prev_is_red = prev["close"] < prev["open"]
            if (is_green and prev_is_red and body > prev_body * 1.2
                    and last["close"] > prev["open"] and last["open"] < prev["close"]):
                patterns_detected.append("看涨吞没")
                buy_signals.append(0.75)
            elif (not is_green and not prev_is_red and body > prev_body * 1.2
                  and last["close"] < prev["open"] and last["open"] > prev["close"]):
                patterns_detected.append("看跌吞没")
                sell_signals.append(0.75)

        # ---- 3. 十字星 ----
        if shadow_ratio_body < 0.1 and upper_shadow > 0 and lower_shadow > 0:
            # 十字星 + 前日趋势 → 反转信号
            if prev is not None and prev["close"] > prev["open"]:  # 上涨后出现
                patterns_detected.append("十字星（高位，看跌反转）")
                sell_signals.append(0.55)
            elif prev is not None and prev["close"] < prev["open"]:  # 下跌后出现
                patterns_detected.append("十字星（低位，看涨反转）")
                buy_signals.append(0.55)

        # ---- 4. 启明星（3K线底部反转） ----
        if prev is not None and prev2 is not None:
            if (prev2["close"] < prev2["open"]  # 大阴线
                    and abs(prev["close"] - prev["open"]) < abs(prev2["close"] - prev2["open"]) * 0.3  # 小实体
                    and is_green and last["close"] > (prev2["open"] + prev2["close"]) / 2):
                patterns_detected.append("启明星（看涨反转）")
                buy_signals.append(0.8)

        # ---- 5. 黄昏星（3K线顶部反转） ----
        if prev is not None and prev2 is not None:
            if (prev2["close"] > prev2["open"]  # 大阳线
                    and abs(prev["close"] - prev["open"]) < abs(prev2["close"] - prev2["open"]) * 0.3  # 小实体
                    and not is_green and last["close"] < (prev2["open"] + prev2["close"]) / 2):
                patterns_detected.append("黄昏星（看跌反转）")
                sell_signals.append(0.8)

        # ---- 6. 大阳线 / 大阴线 ----
        avg_body = abs(df["close"] - df["open"]).rolling(20).mean().iloc[-1] if len(df) > 20 else body
        if avg_body > 0 and body > avg_body * 2:
            if is_green:
                patterns_detected.append("大阳线（强势）")
                buy_signals.append(0.6)
            else:
                patterns_detected.append("大阴线（弱势）")
                sell_signals.append(0.6)

        # ---- 7. 长上影线（压力） ----
        if upper_shadow > body * 2 and upper_shadow > lower_shadow:
            if not is_green:
                patterns_detected.append("长上影线（抛压沉重）")
                sell_signals.append(0.65)

        # ---- 8. 长下影线（支撑） ----
        if lower_shadow > body * 2 and lower_shadow > upper_shadow:
            if is_green:
                patterns_detected.append("长下影线（买盘介入）")
                buy_signals.append(0.65)

        # ---- 综合判断 ----
        avg_buy_conf = np.mean(buy_signals) if buy_signals else 0
        avg_sell_conf = np.mean(sell_signals) if sell_signals else 0

        if avg_buy_conf > avg_sell_conf and avg_buy_conf > 0.5:
            signal = SIGNAL_BUY
            confidence = avg_buy_conf
        elif avg_sell_conf > avg_buy_conf and avg_sell_conf > 0.5:
            signal = SIGNAL_SELL
            confidence = avg_sell_conf
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = f"识别到 {len(patterns_detected)} 个形态: {', '.join(patterns_detected)}" if patterns_detected else "无明确K线形态信号"

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=min(confidence, 1.0),
            strength=len(patterns_detected) / 4.0,
            explanation=explanation,
            patterns_detected=patterns_detected,
        )
