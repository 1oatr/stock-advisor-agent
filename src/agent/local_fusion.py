"""agent/local_fusion.py — LLM 关闭时的本地增强策略引擎

替代 DeepSeek LLM 的深度推理角色。
输入：11 个技能的原始结果 + 6 条规则结果 + 大盘状态 + 技术指标快照
输出：{action, confidence, analysis_text, key_signals, risk_note, predictions}

核心策略（5 维度评分卡）：
1. 信号簇指数加分 —— 多个技能共振 > 单个技能极端值
2. 矛盾信号检测与惩罚 —— 多空信号共存时降权
3. 规则引擎交叉验证 —— 规则和技能互相印证/矛盾时调整置信度
4. 大盘状态限仓 —— 熊市强行压降置信度
5. 统计预测 —— 历史相似模式推导短/长期走势（调用 pattern_cache）
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from src.skills.base import SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD

logger = logging.getLogger(__name__)

# 信号簇配置（可被 LocalFusionEngine(config=...) 覆盖，默认值供配置层复用）
LOCAL_FUSION_DEFAULTS = {
    "cluster_base": 0.25,           # 单个信号的基础分数
    "cluster_multiplier": 1.45,     # 每多一个信号的指数乘数（共振强度）
    "contradiction_penalty": 0.55,  # 有矛盾信号时的折扣系数
    "rule_agree_bonus": 1.20,       # 规则与技能一致 → 置信度×1.20
    "rule_disagree_penalty": 0.55,  # 规则与技能相反 → 置信度×0.55
    "bear_cap": 0.60,               # 熊市最高置信度
    "min_confidence": 0.35,         # 最终置信度低于此值 → hold
    "confidence_threshold": 0.0,    # 技能信号置信度过滤（低于此值的 buy/sell 不参与打分）
}

# 兼容旧引用（模块级常量）
CLUSTER_BASE = LOCAL_FUSION_DEFAULTS["cluster_base"]
CLUSTER_MULTIPLIER = LOCAL_FUSION_DEFAULTS["cluster_multiplier"]
CONTRADICTION_PENALTY = LOCAL_FUSION_DEFAULTS["contradiction_penalty"]
RULE_CROSS_VALIDATION_AGREE = LOCAL_FUSION_DEFAULTS["rule_agree_bonus"]
RULE_CROSS_VALIDATION_DISAGREE = LOCAL_FUSION_DEFAULTS["rule_disagree_penalty"]
BEAR_MARKET_CONFIDENCE_CAP = LOCAL_FUSION_DEFAULTS["bear_cap"]
MIN_CONFIDENCE_FOR_SIGNAL = LOCAL_FUSION_DEFAULTS["min_confidence"]


class LocalFusionEngine:
    """本地增强策略引擎

    用法：
        engine = LocalFusionEngine()
        result = engine.analyze(
            skill_results=skill_results,
            rule_decision=rule_decision,
            market_state="bull",
            indicators_snapshot={...},
            df=df,  # 用于统计预测
            code="600519",
        )
    """

    def __init__(self, config: Optional[dict] = None):
        self.analysis_count = 0
        self._cfg = {**LOCAL_FUSION_DEFAULTS, **(config or {})}

    # =====================================================================
    # 主入口
    # =====================================================================

    def analyze(
        self,
        skill_results: List[SkillResult],
        rule_decision: dict,
        market_state: str = "range",
        indicators_snapshot: Optional[dict] = None,
        df=None,
        code: str = "",
        fetcher=None,
    ) -> dict:
        """一站式本地分析，输出与 LLM 相同格式的决策

        Returns:
            {
                "action": "buy"/"sell"/"hold",
                "confidence": 0.72,
                "analysis_text": "详细分析说明",
                "key_signals": [...],
                "risk_note": "...",
                "predictions": {short_term: {...}, long_term: {...}},
                "source": "local_fusion",
            }
        """
        self.analysis_count += 1

        # ---- 1. 信号聚类评分 ----
        buy_score, sell_score, skill_consensus_label = self._cluster_score(skill_results)

        # ---- 2. 矛盾检测 ----
        contradiction_factor = self._contradiction_factor(skill_results)
        if contradiction_factor < 1.0:
            buy_score *= contradiction_factor
            sell_score *= contradiction_factor

        # ---- 3. 规则交叉验证 ----
        rule_adj = self._rule_cross_validation(buy_score, sell_score, rule_decision)
        buy_score *= rule_adj["buy_factor"]
        sell_score *= rule_adj["sell_factor"]

        # ---- 4. 大盘状态限仓 ----
        market_cap = self._market_confidence_cap(market_state)

        # ---- 5. 确定最终信号 ----
        action, raw_confidence = self._determine_action(buy_score, sell_score, skill_results)
        confidence = min(raw_confidence, market_cap)

        # 如果最小置信度都不够
        if confidence < self._cfg["min_confidence"]:
            action = SIGNAL_HOLD
            confidence = 0.40

        # ---- 6. 生成分析文本 ----
        analysis_text = self._generate_analysis(
            skill_results, rule_decision, market_state,
            buy_score, sell_score, action, confidence, contradiction_factor,
        )

        # ---- 7. 提取关键信号 ----
        key_signals = self._extract_key_signals(skill_results, rule_decision)

        # ---- 8. 风险提示 ----
        risk_note = self._generate_risk_note(
            skill_results, rule_decision, market_state,
            contradiction_factor, action,
        )

        # ---- 9. 统计预测 ----
        predictions = self._statistical_predict(df, code, fetcher)

        return {
            "action": action,
            "confidence": round(confidence, 2),
            "analysis_text": analysis_text,
            "key_signals": key_signals,
            "risk_note": risk_note,
            "predictions": predictions,
            "code": code,
            "source": "local_fusion",
            "_debug": {
                "buy_score": round(buy_score, 3),
                "sell_score": round(sell_score, 3),
                "contradiction_factor": round(contradiction_factor, 3),
                "rule_adj": rule_adj,
                "market_cap": market_cap,
                "consensus": skill_consensus_label,
            },
        }

    # =====================================================================
    # 维度 1：信号簇指数加分
    # =====================================================================

    def _cluster_score(self, skill_results: List[SkillResult]) -> tuple:
        """信号簇指数加分

        核心思想：3 个技能看多 > 1 个技能极度看多。
        不是线性相加，而是簇内信号数越多 → 指数级加分。

        Returns:
            (buy_score, sell_score, consensus_label)
        """
        buy_signals = []
        sell_signals = []

        conf_thr = self._cfg.get("confidence_threshold", 0.0)

        for r in skill_results:
            # 置信度低于阈值的方向信号不参与打分（视为弱信号）
            if (conf_thr > 0.0
                    and r.signal in (SIGNAL_BUY, SIGNAL_SELL, "strong_buy", "strong_sell")
                    and r.confidence < conf_thr):
                continue
            w = r.confidence * (r.strength + 0.3)
            if r.signal in (SIGNAL_BUY, "strong_buy"):
                bonus = 1.3 if r.signal == "strong_buy" else 1.0
                buy_signals.append(w * bonus)
            elif r.signal in (SIGNAL_SELL, "strong_sell"):
                bonus = 1.3 if r.signal == "strong_sell" else 1.0
                sell_signals.append(w * bonus)

        n_buy = len(buy_signals)
        n_sell = len(sell_signals)

        # 指数级簇加分
        buy_score = self._cfg["cluster_base"] * (self._cfg["cluster_multiplier"] ** (n_buy - 1)) if n_buy > 0 else 0.0
        sell_score = self._cfg["cluster_base"] * (self._cfg["cluster_multiplier"] ** (n_sell - 1)) if n_sell > 0 else 0.0

        # 叠加信号强度
        if n_buy > 0:
            avg_buy_strength = np.mean(buy_signals) if buy_signals else 0.5
            buy_score *= (0.5 + avg_buy_strength)
        if n_sell > 0:
            avg_sell_strength = np.mean(sell_signals) if sell_signals else 0.5
            sell_score *= (0.5 + avg_sell_strength)

        # 共识标签
        total_signals = n_buy + n_sell
        if total_signals == 0:
            consensus_label = "no_signal"
        elif n_sell == 0:
            consensus_label = "all_buy"
        elif n_buy == 0:
            consensus_label = "all_sell"
        elif n_buy >= n_sell * 3:
            consensus_label = "strong_buy_majority"
        elif n_sell >= n_buy * 3:
            consensus_label = "strong_sell_majority"
        elif n_buy > n_sell:
            consensus_label = "mixed_buy_bias"
        elif n_sell > n_buy:
            consensus_label = "mixed_sell_bias"
        else:
            consensus_label = "balanced"

        return buy_score, sell_score, consensus_label

    # =====================================================================
    # 维度 2：矛盾信号惩罚
    # =====================================================================

    def _contradiction_factor(self, skill_results: List[SkillResult]) -> float:
        """矛盾信号惩罚系数

        当买入和卖出信号同时存在时，对双方的打分都降权。
        矛盾越激烈（买/卖数量接近），降权越多。
        """
        n_buy = sum(1 for r in skill_results
                    if r.signal in (SIGNAL_BUY, "strong_buy"))
        n_sell = sum(1 for r in skill_results
                     if r.signal in (SIGNAL_SELL, "strong_sell"))

        if n_buy == 0 or n_sell == 0:
            return 1.0  # 无矛盾

        # 矛盾比例：min/max，范围 (0, 1]
        # 1买 vs 3卖 → 1/3=0.33 → 轻罚
        # 2买 vs 2卖 → 2/2=1.0 → 重罚
        ratio = min(n_buy, n_sell) / max(n_buy, n_sell)

        # 惩罚 = 1 - (比例 × (1 - penalty))
        factor = 1.0 - ratio * (1.0 - self._cfg["contradiction_penalty"])

        return max(factor, 0.30)  # 不低于 0.30

    # =====================================================================
    # 维度 3：规则引擎交叉验证
    # =====================================================================

    def _rule_cross_validation(self, buy_score: float, sell_score: float,
                                rule_decision: dict) -> dict:
        """规则引擎交叉验证

        如果技能共识和规则引擎结论一致 → 置信度加成
        如果相反 → 置信度大降（规则是客观的，有矛盾时要谨慎）
        """
        rule_signal = rule_decision.get("signal", "hold")
        rule_strength = rule_decision.get("strength", 0.3)

        buy_factor = 1.0
        sell_factor = 1.0

        if rule_signal == "buy":
            buy_factor *= self._cfg["rule_agree_bonus"]
            sell_factor *= self._cfg["rule_disagree_penalty"]
        elif rule_signal == "sell":
            sell_factor *= self._cfg["rule_agree_bonus"]
            buy_factor *= self._cfg["rule_disagree_penalty"]
        # hold → 不做调整（规则也看不清）

        # 规则强度弱时减轻调整幅度
        if rule_strength < 0.4:
            buy_factor = 1.0 + (buy_factor - 1.0) * 0.5
            sell_factor = 1.0 + (sell_factor - 1.0) * 0.5

        return {"buy_factor": round(buy_factor, 3), "sell_factor": round(sell_factor, 3)}

    # =====================================================================
    # 维度 4：大盘状态限仓
    # =====================================================================

    def _market_confidence_cap(self, market_state: str) -> float:
        """大盘状态 → 置信度上限

        熊市中即使技能全看多，置信度也有上限。
        """
        caps = {
            "bull": 1.0,
            "range": 0.85,
            "bear": self._cfg["bear_cap"],
        }
        return caps.get(market_state, 0.80)

    # =====================================================================
    # 维度 5：确定最终信号
    # =====================================================================

    def _determine_action(self, buy_score: float, sell_score: float,
                          skill_results: List[SkillResult]) -> tuple:
        """确定最终动作和置信度"""
        n_buy = sum(1 for r in skill_results
                    if r.signal in (SIGNAL_BUY, "strong_buy"))
        n_sell = sum(1 for r in skill_results
                     if r.signal in (SIGNAL_SELL, "strong_sell"))

        score_diff = buy_score - sell_score

        if score_diff > 0.05:
            action = SIGNAL_BUY
            # 置信度 = 买入分占比 + 买入信号数加成
            total_for_action = buy_score / (buy_score + sell_score + 1e-8)
            n_bonus = min((n_buy - n_sell) * 0.05, 0.20) if n_buy > n_sell else 0
            confidence = min(total_for_action + n_bonus, 1.0)

        elif score_diff < -0.05:
            action = SIGNAL_SELL
            total_for_action = sell_score / (buy_score + sell_score + 1e-8)
            n_bonus = min((n_sell - n_buy) * 0.05, 0.20) if n_sell > n_buy else 0
            confidence = min(total_for_action + n_bonus, 1.0)

        else:
            action = SIGNAL_HOLD
            confidence = 0.40

        return action, confidence

    # =====================================================================
    # 分析文本生成
    # =====================================================================

    def _generate_analysis(self, skill_results: List[SkillResult],
                           rule_decision: dict, market_state: str,
                           buy_score: float, sell_score: float,
                           action: str, confidence: float,
                           contradiction_factor: float) -> str:
        """生成与 LLM 风格一致的自然语言分析"""
        n_buy = sum(1 for r in skill_results
                    if r.signal in (SIGNAL_BUY, "strong_buy"))
        n_sell = sum(1 for r in skill_results
                     if r.signal in (SIGNAL_SELL, "strong_sell"))
        n_hold = len(skill_results) - n_buy - n_sell

        rule_signal = rule_decision.get("signal", "hold")

        parts = []

        # 信号概况
        parts.append(f"共{len(skill_results)}个技能：{n_buy}看多、{n_sell}看空、{n_hold}中性")

        # 共识度
        if n_buy >= n_sell * 2:
            parts.append("多头信号占主导，一致性较高")
        elif n_sell >= n_buy * 2:
            parts.append("空头信号占主导，一致性较高")
        elif n_buy == n_sell:
            parts.append("多空均衡，方向不明")
        else:
            parts.append(f"信号存在分歧（{n_buy}买 vs {n_sell}卖）")

        # 矛盾情况
        if contradiction_factor < 0.8:
            parts.append("部分技能信号互相矛盾，已降权处理")

        # 规则验证
        if rule_signal == action:
            parts.append(f"硬规则引擎（{rule_signal}）与技能共识一致，增强可信度")
        elif rule_signal != "hold" and action != "hold":
            parts.append(f"⚠️ 硬规则引擎偏{rule_signal}，与技能共识({action})相左，需谨慎")

        # 最终结论
        action_label = {"buy": "增持", "sell": "减持", "hold": "持有"}
        strength_label = "强烈" if confidence >= 0.70 else ("适度" if confidence >= 0.50 else "轻度")
        market_labels = {"bull": "牛市", "bear": "熊市", "range": "震荡市"}
        parts.append(
            f"综合判断：{strength_label}{action_label.get(action, '持有')}"
            f"（{market_labels.get(market_state, '未知市场')}环境下）"
        )

        return "；".join(parts)

    # =====================================================================
    # 关键信号提取
    # =====================================================================

    def _extract_key_signals(self, skill_results: List[SkillResult],
                             rule_decision: dict) -> List[str]:
        """从技能结果中提取最重要的买入/卖出信号"""
        signals = []

        for r in skill_results:
            if r.signal in (SIGNAL_BUY, "strong_buy", SIGNAL_SELL, "strong_sell"):
                # 只取置信度 > 0.4 的信号
                if r.confidence > 0.4 and r.patterns_detected:
                    for pattern in r.patterns_detected[:2]:
                        prefix = {
                            SIGNAL_BUY: "📈",
                            "strong_buy": "📈📈",
                            SIGNAL_SELL: "📉",
                            "strong_sell": "📉📉",
                        }.get(r.signal, "")
                        signals.append(f"{prefix}[{r.skill_name}] {pattern}")

        # 加入规则信号
        rule_top = rule_decision.get("top_rules", [])[:2]
        for rt in rule_top:
            signals.append(f"📐[规则] {rt.get('name', '')}: {rt.get('explanation', '')}")

        return signals[:8]  # 最多 8 条

    # =====================================================================
    # 风险提示
    # =====================================================================

    def _generate_risk_note(self, skill_results: List[SkillResult],
                            rule_decision: dict, market_state: str,
                            contradiction_factor: float, action: str) -> str:
        """生成风险提示"""
        risks = []

        if market_state == "bear":
            risks.append("大盘处于熊市，任何买入操作风险都高于正常水平")
        elif market_state == "range":
            risks.append("大盘处于震荡市，方向不明朗，建议控制仓位")

        if contradiction_factor < 0.8:
            risks.append("多空信号存在矛盾，结论可靠性下降")

        # 检查波动率技能
        for r in skill_results:
            if r.skill_name == "volatility_regime" and r.metadata:
                regime = r.metadata.get("regime", "")
                if regime == "high_vol_panic":
                    risks.append("当前处于高波区间，不建议建仓或加仓")
                elif regime == "low_vol_squeeze":
                    risks.append("波动率极低，变盘在即，注意方向选择风险")

        # 检查多时间框架
        for r in skill_results:
            if r.skill_name == "multi_timeframe_consensus":
                meta = r.metadata or {}
                if meta.get("consensus") == "none":
                    risks.append("日/周/月线趋势不一致，中短期走势可能反复")

        if rule_decision.get("signal") == "sell" and action == "buy":
            risks.append("硬规则看空但技能看多，建议等待规则确认后再入场")

        if not risks:
            risks.append("当前无特别风险提示，常规交易风险可控")

        return "；".join(risks)

    # =====================================================================
    # 统计预测
    # =====================================================================

    def _statistical_predict(self, df, code: str, fetcher=None) -> dict:
        """调用 pattern_cache 做统计预测"""
        if df is None or df.empty:
            return self._empty_prediction("无数据")

        try:
            from src.data.pattern_cache import predict_from_patterns
            return predict_from_patterns(df, code=code, fetcher=fetcher)
        except Exception as e:
            logger.warning(f"统计预测失败: {e}")
            return self._empty_prediction(f"统计预测异常: {str(e)[:50]}")

    def _empty_prediction(self, reason: str) -> dict:
        return {
            "short_term": {
                "days": "3-5天",
                "direction": "震荡",
                "probability": "小概率",
                "change_pct": 0.0,
                "reason": reason,
            },
            "mid_term": {
                "days": "5-15天",
                "direction": "震荡",
                "probability": "小概率",
                "change_pct": 0.0,
                "reason": reason,
            },
            "mid_long_term": {
                "days": "15-40天",
                "direction": "震荡",
                "probability": "小概率",
                "change_pct": 0.0,
                "reason": reason,
            },
            "long_term": {
                "days": "40-80天",
                "direction": "震荡",
                "probability": "小概率",
                "change_pct": 0.0,
                "reason": reason,
            },
        }
