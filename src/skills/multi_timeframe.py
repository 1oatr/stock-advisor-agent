"""skills/multi_timeframe.py — 多时间框架一致性技能

对日线数据做周线/月线级别的降采样分析，检查多级趋势是否一致。

核心逻辑：
- 日月周三级共振向上 → 最可靠的买入信号
- 日月周三级共振向下 → 最可靠的卖出信号
- 日线向上但周线向下 → 短期反弹不持久，谨慎追涨
- 日线向下但周线向上 → 短期回调不恐慌，可能是买点

这是现有 8 个技能的重要补充——它们都只看日线级别。
"""

import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class MultiTimeframeSkill(BaseSkill):
    """多时间框架一致性检查

    降采样：
    - 周线：从日线按周聚合（取周五或最后交易日）
    - 月线：从日线按月聚合（取月最后一个交易日）
    """

    def __init__(self):
        super().__init__(
            name="multi_timeframe_consensus",
            description="检查日/周/月线趋势一致性，过滤单级别假信号",
            category="趋势分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 40:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.0, explanation="数据不足（需≥40个交易日）",
            )

        # 确保有 date 列用于降采样
        if "date" not in df.columns:
            # 无日期列时，用索引替代（假定按时间排序）
            df = df.copy()
            df["_idx"] = range(len(df))

        patterns = []

        # ---- 1. 日线级别趋势 ----
        daily_trend = self._assess_trend(df, level="日线")
        patterns.append(f"日线: {daily_trend['label']}")

        # ---- 2. 周线级别趋势 ----
        weekly_df = self._resample_to_weekly(df)
        weekly_trend = self._assess_trend(weekly_df, level="周线")
        patterns.append(f"周线: {weekly_trend['label']}")

        # ---- 3. 月线级别趋势 ----
        monthly_df = self._resample_to_monthly(df)
        monthly_trend = self._assess_trend(monthly_df, level="月线")
        patterns.append(f"月线: {monthly_trend['label']}")

        # ---- 4. 一致性判断 ----
        trends = [daily_trend, weekly_trend, monthly_trend]
        up_count = sum(1 for t in trends if t["direction"] == "up")
        down_count = sum(1 for t in trends if t["direction"] == "down")

        buy_conf = 0.0
        sell_conf = 0.0

        if up_count == 3:
            # 三级共振向上 — 最强买入信号
            patterns.append("日/周/月三级共振向上，趋势可靠性极高")
            buy_conf = 0.75
            signal = SIGNAL_BUY

        elif down_count == 3:
            # 三级共振向下 — 最强卖出信号
            patterns.append("日/周/月三级共振向下，下降趋势明确")
            sell_conf = 0.75
            signal = SIGNAL_SELL

        elif up_count == 2 and down_count == 1:
            disagree = [t for t in trends if t["direction"] == "down"][0]
            patterns.append(f"两级看多但{disagree['level']}偏空，整体偏多但需留意")
            buy_conf = 0.45

            # 日线多 + 周线空 = 短期反弹 vs 中期下跌
            if daily_trend["direction"] == "up" and weekly_trend["direction"] == "down":
                patterns.append("⚠️ 日线反弹 vs 周线下跌，反弹高度受限，不宜追涨")
                buy_conf *= 0.6
            signal = SIGNAL_BUY if buy_conf > 0.25 else SIGNAL_HOLD

        elif down_count == 2 and up_count == 1:
            disagree = [t for t in trends if t["direction"] == "up"][0]
            patterns.append(f"两级看空但{disagree['level']}偏多，整体偏空")
            sell_conf = 0.45

            # 日线空 + 周线多 = 短期回调 vs 中期上升
            if daily_trend["direction"] == "down" and weekly_trend["direction"] == "up":
                patterns.append("💡 日线回调 vs 周线上升，回调可能是买入机会")
                sell_conf *= 0.5
                buy_conf += 0.25
            signal = SIGNAL_SELL if sell_conf > 0.25 else SIGNAL_HOLD

        else:
            patterns.append("各级别趋势不明朗，建议持有")
            signal = SIGNAL_HOLD

        # ---- 5. 趋势强度 ----
        # 日线 MA 发散度：发散越大 = 趋势越强
        daily_strength = self._trend_divergence(df)

        confidence = max(buy_conf, sell_conf, 0.45)
        if signal != SIGNAL_HOLD:
            confidence = min(confidence * daily_strength, 1.0)

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=(up_count if signal == SIGNAL_BUY else
                      down_count if signal == SIGNAL_SELL else 1) / 3.0,
            explanation="; ".join(patterns),
            patterns_detected=patterns,
            metadata={
                "daily_trend": daily_trend,
                "weekly_trend": weekly_trend,
                "monthly_trend": monthly_trend,
                "up_count": up_count,
                "down_count": down_count,
                "consensus": "full" if up_count == 3 or down_count == 3 else
                            "partial" if up_count == 2 or down_count == 2 else "none",
            },
        )

    # =====================================================================
    # 内部方法
    # =====================================================================

    def _assess_trend(self, df: pd.DataFrame, level: str = "") -> dict:
        """评估单个时间框架的趋势

        Returns:
            {"direction": "up"/"down"/"neutral", "label": "上升/下降/横盘",
             "strength": 0~1, "details": [...]}
        """
        if len(df) < 2:
            return {"direction": "neutral", "label": "数据不足", "strength": 0.3}

        close = df["close"]

        # MA5 和 MA20（如果列存在则直接取，否则计算）
        ma5 = df["MA5"].iloc[-1] if "MA5" in df.columns else close.rolling(5).mean().iloc[-1]
        ma20 = df["MA20"].iloc[-1] if "MA20" in df.columns else close.rolling(20).mean().iloc[-1]
        current = close.iloc[-1]

        # MA20 斜率（近 5 根/周/月 K 线的 MA20 趋势）
        if len(df) >= 5:
            ma20_series = (df["MA20"] if "MA20" in df.columns
                          else close.rolling(20).mean())
            ma20_slope = (ma20_series.iloc[-1] / (ma20_series.iloc[-5] + 1e-8) - 1) * 100
        else:
            ma20_slope = 0

        # 趋势判断
        details = []

        if current > ma5 > ma20 and ma20_slope > 0.5:
            direction = "up"
            label = "上升趋势（多头排列）"
            strength = min(0.8, 0.5 + abs(ma20_slope) / 20)
            details.append(f"价格({current:.2f})>MA5({ma5:.2f})>MA20({ma20:.2f})")
        elif current < ma5 < ma20 and ma20_slope < -0.5:
            direction = "down"
            label = "下降趋势（空头排列）"
            strength = min(0.8, 0.5 + abs(ma20_slope) / 20)
            details.append(f"价格({current:.2f})<MA5({ma5:.2f})<MA20({ma20:.2f})")
        elif current > ma5 and ma20_slope > 0:
            direction = "up"
            label = "偏强（站稳MA5，MA20向上）"
            strength = 0.55
        elif current < ma5 and ma20_slope < 0:
            direction = "down"
            label = "偏弱（跌破MA5，MA20向下）"
            strength = 0.55
        else:
            direction = "neutral"
            label = "横盘整理"
            strength = 0.40

        return {
            "direction": direction,
            "label": label,
            "level": level,
            "strength": strength,
            "ma20_slope_pct": round(ma20_slope, 2),
            "details": details,
        }

    def _resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线 → 周线（按周聚合）

        每根周线K线 = 本周所有日线的聚合：
        - open = 周一开盘
        - close = 周五收盘（或本周最后一天）
        - high = 本周最高
        - low = 本周最低
        - volume = 本周成交量合计
        """
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df_weekly = df.set_index("date").resample("W").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
        else:
            # 无日期列：每5个交易日一组
            n = (len(df) // 5) * 5
            if n < 5:
                return df.tail(1)
            groups = df.iloc[-n:].copy()
            groups["_w"] = range(n // 5)
            records = []
            offset = len(df) - n
            for i in range(n // 5):
                chunk = groups.iloc[i*5:(i+1)*5]
                records.append({
                    "open": chunk["open"].iloc[0],
                    "high": chunk["high"].max(),
                    "low": chunk["low"].min(),
                    "close": chunk["close"].iloc[-1],
                    "volume": chunk["volume"].sum(),
                })
            df_weekly = pd.DataFrame(records)

        return df_weekly

    def _resample_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """日线 → 月线"""
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df_monthly = df.set_index("date").resample("ME").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
        else:
            # 无日期列：每20个交易日一组
            n = (len(df) // 20) * 20
            if n < 20:
                return df.tail(1)
            groups = df.iloc[-n:].copy()
            groups["_m"] = [i // 20 for i in range(n)]
            records = []
            for i in range(n // 20):
                chunk = groups.iloc[i*20:(i+1)*20]
                records.append({
                    "open": chunk["open"].iloc[0],
                    "high": chunk["high"].max(),
                    "low": chunk["low"].min(),
                    "close": chunk["close"].iloc[-1],
                    "volume": chunk["volume"].sum(),
                })
            df_monthly = pd.DataFrame(records)

        return df_monthly

    def _trend_divergence(self, df: pd.DataFrame) -> float:
        """计算日线均线发散度（趋势强度指标）

        发散度越高 = 趋势越强（不管是涨还是跌）
        发散度越低 = 均线缠绕、方向不明
        """
        if len(df) < 20:
            return 0.8

        ma5 = df["MA5"].iloc[-1] if "MA5" in df.columns else df["close"].rolling(5).mean().iloc[-1]
        ma20 = df["MA20"].iloc[-1] if "MA20" in df.columns else df["close"].rolling(20).mean().iloc[-1]
        ma60 = df["MA60"].iloc[-1] if "MA60" in df.columns else df["close"].rolling(60).mean().iloc[-1]
        close = df["close"].iloc[-1]

        if close == 0:
            return 0.8

        mas = [ma5, ma20, ma60]
        if 0 in mas:
            return 0.8

        spread = max(mas) / min(mas) - 1  # 最大均线差距比例
        # 发散度 1-5% → 强趋势，>10% → 极度发散
        if spread > 0.10:
            return 1.15  # 均线极度发散，趋势强
        elif spread > 0.05:
            return 1.05
        elif spread < 0.02:
            return 0.85  # 均线缠绕，趋势模糊
        else:
            return 1.0
