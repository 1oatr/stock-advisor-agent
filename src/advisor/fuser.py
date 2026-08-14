"""advisor/fuser.py — 决策融合层

LLM+Skills(40%) + RL智能体(30%) + 硬规则(30%) → 三方动态加权融合，生成最终买卖决策。
"""

from typing import Dict, List, Optional, Tuple


class DecisionFuser:
    """决策融合器

    将三个信号源加权融合：
    1. LLM+Skills 深度分析（权重最高 — 理解全局关联）
    2. RL 智能体（历史数据训练出的买卖策略）
    3. 硬规则引擎（6条技术规则）

    权重根据近期准确率动态调整。
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None, adapt_window: int = 20):
        self.weights = initial_weights or {
            "llm_skills": 0.40,
            "rl": 0.30,
            "rule": 0.30,
        }
        self.adapt_window = adapt_window
        self.performance_history: List[dict] = []

    def fuse(
        self,
        rl_decision: Dict,
        rule_decision: Dict,
        llm_skills_decision: Dict = None,
        skill_decision: Dict = None,
        market_state: str = "unknown",
        rl_untrained: bool = False,
    ) -> Dict:
        """融合 LLM+Skills + RL + 规则 三方决策

        Args:
            rl_decision: {"action": "buy"/"sell"/"hold", "confidence": 0.72}
            rule_decision: {"signal": "buy"/"sell"/"hold", "strength": 0.68, "consensus": 0.6}
            llm_skills_decision: {"action": "buy"/"sell"/"hold", "confidence": 0.72, ...}
            skill_decision: (向后兼容) 同 llm_skills_decision
            market_state: 市场状态

        Returns:
            {
                "action": "buy",
                "confidence": 0.74,
                "position": 0.15,
                "llm_score": 0.72,
                "rl_score": 0.60,
                "rule_score": 0.68,
                "fusion_score": 0.70,
                "consensus_level": "high",
                "signals": {...},
            }
        """
        # 向后兼容：skill_decision 等同于 llm_skills_decision
        llm = llm_skills_decision or skill_decision or {}

        # 提取各信号
        llm_action = llm.get("action", llm.get("signal", "hold"))
        llm_conf = llm.get("confidence", 0.5)

        rl_action = rl_decision.get("action", "hold")
        rl_conf = rl_decision.get("confidence", 0.5)

        rule_signal = rule_decision.get("signal", "hold")
        rule_strength = rule_decision.get("strength", 0.5)

        # 计算三方共识度
        consensus_level, is_consensus = self._evaluate_consensus(
            llm_action, rl_action, rule_signal
        )

        # 计算融合得分
        action_scores = self._calculate_action_scores(
            llm_action, llm_conf,
            rl_action, rl_conf,
            rule_signal, rule_strength,
        )

        # 确定最终决策
        final_action = max(action_scores, key=action_scores.get)
        fusion_score = action_scores[final_action]

        # 计算置信度（考虑共识度加成）
        consensus_bonus = 0.1 if is_consensus else -0.1
        confidence = min(max(fusion_score + consensus_bonus, 0), 1)

        # 仓位建议（凯利公式）
        # RL 未训练时共识度降一级：少一路信号，不确定性更高
        eff_consensus = consensus_level
        if rl_untrained and consensus_level == "high":
            eff_consensus = "medium"
        elif rl_untrained and consensus_level == "medium":
            eff_consensus = "low"
        elif rl_untrained and consensus_level == "low":
            eff_consensus = "low"  # 已经最低

        position = self._suggest_position(
            final_action, confidence, is_consensus,
            consensus_level=eff_consensus, market_state=market_state,
        )

        return {
            "action": final_action,
            "confidence": round(confidence, 4),
            "position": position,
            "llm_score": llm_conf,
            "rl_score": rl_conf,
            "rule_score": rule_strength,
            "fusion_score": round(fusion_score, 4),
            "consensus_level": consensus_level,
            "is_consensus": is_consensus,
            "signals": {
                "llm_skills": {"action": llm_action, "confidence": llm_conf},
                "rl": {"action": rl_action, "confidence": rl_conf},
                "rule": {"signal": rule_signal, "strength": rule_strength},
            },
        }

    def fuse_multi(
        self,
        rl_decisions: List[Dict],
        rule_decisions: List[Dict],
        llm_decisions: List[Dict],
        market_state: str,
    ) -> List[Dict]:
        """批量融合多只股票"""
        results = []
        for i in range(len(llm_decisions or rl_decisions)):
            rl = rl_decisions[i] if i < len(rl_decisions) else {}
            rule = rule_decisions[i] if i < len(rule_decisions) else {}
            llm = (llm_decisions or [])[i] if i < len(llm_decisions or []) else {}
            result = self.fuse(rl, rule, llm_skills_decision=llm, market_state=market_state)
            results.append(result)
        return results

    def update_weights(self, recent_accuracy: Dict[str, float]):
        """根据近期准确率动态调整三方权重

        Args:
            recent_accuracy: {"llm_skills": 0.72, "rl": 0.65, "rule": 0.68}
        """
        total = sum(recent_accuracy.values())
        if total > 0:
            for key in self.weights:
                if key in recent_accuracy:
                    self.weights[key] = recent_accuracy[key] / total

    def _calculate_action_scores(
        self, llm_action: str, llm_conf: float,
        rl_action: str, rl_conf: float,
        rule_signal: str, rule_strength: float,
    ) -> Dict[str, float]:
        """计算各动作的加权得分"""
        scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

        # LLM+Skills 贡献 (40%)
        if llm_action in scores:
            scores[llm_action] += llm_conf * self.weights.get("llm_skills", 0.40)

        # RL 贡献 (30%)
        if rl_action in scores:
            scores[rl_action] += rl_conf * self.weights.get("rl", 0.30)

        # 规则贡献 (30%)
        if rule_signal in scores:
            scores[rule_signal] += rule_strength * self.weights.get("rule", 0.30)

        return scores

    def _evaluate_consensus(self, signal_a: str, signal_b: str, signal_c: str) -> Tuple[str, bool]:
        """评估三方共识度"""
        signals = [signal_a, signal_b, signal_c]
        unique = set(signals)

        if len(unique) == 1:
            return "high", True
        elif len(unique) == 2:
            return "medium", True
        else:
            return "low", False

    def _suggest_position(self, action: str, confidence: float,
                          is_consensus: bool, consensus_level: str = "medium",
                          market_state: str = "unknown") -> float:
        """凯利公式仓位计算

        f* = (b × p - q) / b  →  actual = f* × k × market_factor

        参数:
            b = 4.0 (配置止盈20% / 止损5%)
            p = confidence (融合置信度)
            k = 0.5/0.35/0.2 (高/中/低共识)
        """
        if action == "hold":
            return 0.0

        from .kelly import get_kelly
        kelly = get_kelly()

        # RL 未训练时共识度降一级（少一路信号，不确定性更高）
        eff_consensus = consensus_level

        return kelly.calculate(
            confidence=confidence,
            consensus=eff_consensus,
            market_state=market_state,
        )

    def _check_consensus(self, rl_action: str, rule_signal: str) -> bool:
        """检查 RL 与规则是否一致（向后兼容）"""
        return rl_action == rule_signal
