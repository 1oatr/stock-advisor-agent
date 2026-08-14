"""skills/fund_flow.py — 主力资金流向分析技能

利用 enrichment.py 已算好的 main_flow_pct / main_flow_5d / turnover_rate，
判断主力是在吸筹、出货还是按兵不动。

A 股是资金驱动市，主力资金流向是最强的领先指标之一。
"""

import pandas as pd
import numpy as np
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class FundFlowSkill(BaseSkill):
    """主力资金流向分析

    分析维度：
    1. 主力净流入占比（当日 + 近5日累计）
    2. 换手率变化 — 放量+主力流入=吸筹，放量+主力流出=出货
    3. 价格 vs 资金流向背离 — 价跌量增+主力流入=打压吸筹
    4. 连续流入/流出天数
    """

    def __init__(self):
        super().__init__(
            name="fund_flow_analysis",
            description="分析主力资金流向和换手率，判断主力吸筹/出货动向",
            category="资金分析",
        )

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        if df.empty or len(df) < 5:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.0, explanation="数据不足",
            )

        patterns = []
        buy_conf = 0.0
        sell_conf = 0.0

        # ---- 检查是否有资金流向数据 ----
        has_flow = "main_flow_pct" in df.columns and "main_flow_5d" in df.columns
        has_turnover = "turnover_rate" in df.columns

        if not has_flow:
            return SkillResult(
                skill_name=self.name, signal=SIGNAL_HOLD,
                confidence=0.0, strength=0.0,
                explanation="缺少资金流向数据 (main_flow_pct/main_flow_5d)，请先运行 enrich_all()",
            )

        last = df.iloc[-1]
        main_flow_pct = float(last["main_flow_pct"])
        main_flow_5d = float(last["main_flow_5d"])

        # 近5日主力净流入
        recent_flows = df["main_flow_pct"].tail(5).values
        recent_flows_clean = [float(f) for f in recent_flows if not np.isnan(f)]

        # 主力连续流入/流出天数
        consecutive_in = 0
        consecutive_out = 0
        for f in reversed(recent_flows_clean):
            if f > 0.05:
                consecutive_in += 1
            else:
                break
        for f in reversed(recent_flows_clean):
            if f < -0.05:
                consecutive_out += 1
            else:
                break

        # ---- 1. 主力大幅净流入 ----
        if main_flow_pct > 2.0:
            price_chg_5d = df["close"].pct_change(5).iloc[-1] * 100 if len(df) > 5 else 0
            # 主力流入 + 股价未大涨 = 吸筹（黄金信号）
            if price_chg_5d < 5.0:
                patterns.append(f"主力大幅流入({main_flow_pct:+.1f}%)但股价未涨，疑似吸筹")
                buy_conf += 0.70
            else:
                # 主力流入 + 股价已大涨 = 可能拉高出货
                patterns.append(f"主力流入({main_flow_pct:+.1f}%)但涨幅已大({price_chg_5d:+.1f}%)，需警惕")
                buy_conf += 0.35

        elif main_flow_pct > 1.0:
            patterns.append(f"主力净流入({main_flow_pct:+.1f}%)，短期偏多")
            buy_conf += 0.45

        # ---- 2. 主力大幅净流出 ----
        if main_flow_pct < -2.0:
            price_chg = df["close"].pct_change().iloc[-1] * 100
            if price_chg < -2.0:
                patterns.append(f"主力大幅流出({main_flow_pct:+.1f}%)伴随放量下跌，明确出货")
                sell_conf += 0.70
            else:
                patterns.append(f"主力大幅流出({main_flow_pct:+.1f}%)，短期偏空")
                sell_conf += 0.50

        elif main_flow_pct < -1.0:
            patterns.append(f"主力净流出({main_flow_pct:+.1f}%)，短期偏空")
            sell_conf += 0.35

        # ---- 3. 主力连续流入/流出 ----
        if consecutive_in >= 3:
            patterns.append(f"主力连续{consecutive_in}日净流入，建仓迹象")
            buy_conf += 0.60

        if consecutive_out >= 3:
            patterns.append(f"主力连续{consecutive_out}日净流出，减持迹象")
            sell_conf += 0.55

        # ---- 4. 5日资金流向累计 ----
        if main_flow_5d > 3.0:
            patterns.append(f"近5日主力累计大幅流入({main_flow_5d:+.1f}%)，中期看多")
            buy_conf += 0.50
        elif main_flow_5d > 1.0:
            patterns.append(f"近5日主力累计流入({main_flow_5d:+.1f}%)，偏多")
            buy_conf += 0.30
        elif main_flow_5d < -3.0:
            patterns.append(f"近5日主力累计大幅流出({main_flow_5d:+.1f}%)，中期看空")
            sell_conf += 0.50
        elif main_flow_5d < -1.0:
            patterns.append(f"近5日主力累计流出({main_flow_5d:+.1f}%)，偏空")
            sell_conf += 0.30

        # ---- 5. 换手率异常 ----
        if has_turnover:
            turnover = float(last["turnover_rate"])
            avg_turnover_20 = df["turnover_rate"].rolling(20).mean().iloc[-1] if len(df) >= 20 else turnover

            if turnover > 5.0 and main_flow_pct > 0.5:
                patterns.append(f"换手率偏高({turnover:.1f}%)+主力流入，活跃吸筹")
                buy_conf += 0.40
            elif turnover > 5.0 and main_flow_pct < -0.5:
                patterns.append(f"换手率偏高({turnover:.1f}%)+主力流出，活跃出货")
                sell_conf += 0.40
            elif turnover < 0.5 and avg_turnover_20 > 1.0:
                patterns.append(f"换手率骤降({turnover:.1f}%)，交投清淡")
                # 缩量不一定是坏信号，需要结合价格判断
                if df["close"].pct_change(5).iloc[-1] < -0.03:
                    patterns.append("缩量下跌（可能洗盘尾声）")
                    buy_conf += 0.25

        # ---- 6. 价格 vs 资金背离 ----
        if len(df) > 5:
            price_5d = df["close"].pct_change(5).iloc[-1] * 100
            avg_flow_5d = np.mean(recent_flows_clean) if recent_flows_clean else 0
            # 价跌 + 主力流入 = 打压吸筹
            if price_5d < -3.0 and avg_flow_5d > 0.5:
                patterns.append(f"价跌({price_5d:+.1f}%) vs 主力流入({avg_flow_5d:+.1f}%)，打压吸筹")
                buy_conf += 0.55
            # 价涨 + 主力流出 = 拉高出货
            if price_5d > 3.0 and avg_flow_5d < -0.5:
                patterns.append(f"价涨({price_5d:+.1f}%) vs 主力流出({avg_flow_5d:+.1f}%)，拉高出货")
                sell_conf += 0.55

        # ---- 综合判断 ----
        if buy_conf > sell_conf and buy_conf > 0.3:
            signal = SIGNAL_BUY
            confidence = min(buy_conf, 1.0)
        elif sell_conf > buy_conf and sell_conf > 0.3:
            signal = SIGNAL_SELL
            confidence = min(sell_conf, 1.0)
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        explanation = (
            f"资金分析: {', '.join(patterns)}"
            if patterns
            else f"主力资金无明显异动（当日净流入{main_flow_pct:+.1f}%，5日累计{main_flow_5d:+.1f}%）"
        )

        return SkillResult(
            skill_name=self.name,
            signal=signal,
            confidence=confidence,
            strength=len(patterns) / 5.0,
            explanation=explanation,
            patterns_detected=patterns,
            metadata={
                "main_flow_pct": main_flow_pct,
                "main_flow_5d": main_flow_5d,
                "consecutive_in": consecutive_in,
                "consecutive_out": consecutive_out,
            },
        )
