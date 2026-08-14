"""advisor/confidence.py — 置信度评估

综合评估推荐的可信度，考虑因素包括：
- RL 模型置信度
- 规则共识度
- RL 与规则的一致性
- 近期预测准确率
"""

from typing import Dict, Optional


class ConfidenceEvaluator:
    """置信度评估器"""

    def __init__(self):
        self.history: list = []

    def evaluate(
        self,
        rl_confidence: float,
        rule_consensus: float,
        is_consensus: bool,
        market_state: str = "unknown",
        rl_accuracy: Optional[float] = None,
        rule_accuracy: Optional[float] = None,
    ) -> Dict:
        """综合评估置信度

        Args:
            rl_confidence: RL 模型置信度 (0~1)
            rule_consensus: 规则共识度 (0~1)
            is_consensus: RL 与规则是否一致
            market_state: 市场状态
            rl_accuracy: RL 近期准确率（可选）
            rule_accuracy: 规则近期准确率（可选）

        Returns:
            {
                "overall_confidence": 0.75,
                "level": "high",
                "factors": {
                    "rl_factor": 0.72,
                    "rule_factor": 0.68,
                    "consensus_bonus": 0.10,
                    "accuracy_bonus": 0.05,
                    "market_penalty": 0.0,
                }
            }
        """
        factors = {}

        # 1. RL 置信度因子
        factors["rl_factor"] = rl_confidence

        # 2. 规则共识度因子
        factors["rule_factor"] = rule_consensus

        # 3. 共识加成
        factors["consensus_bonus"] = 0.10 if is_consensus else -0.10

        # 4. 历史准确率加成
        accuracy_bonus = 0.0
        if rl_accuracy is not None and rule_accuracy is not None:
            avg_accuracy = (rl_accuracy + rule_accuracy) / 2
            accuracy_bonus = (avg_accuracy - 0.5) * 0.2  # 高于50%加分，低则扣分
        factors["accuracy_bonus"] = round(accuracy_bonus, 4)

        # 5. 市场状态惩罚
        market_penalty = self._get_market_penalty(market_state)
        factors["market_penalty"] = market_penalty

        # 计算综合置信度
        overall = (
            factors["rl_factor"] * 0.35
            + factors["rule_factor"] * 0.25
            + factors["consensus_bonus"]
            + factors["accuracy_bonus"]
            - factors["market_penalty"]
        )

        overall = max(0.0, min(1.0, overall))

        # 记录到历史
        self.history.append({
            "overall": overall,
            "factors": factors,
            "market_state": market_state,
        })

        return {
            "overall_confidence": round(overall, 4),
            "level": self._calculate_level(overall),
            "factors": factors,
        }

    def _calculate_level(self, confidence: float) -> str:
        """根据置信度值返回等级标签"""
        if confidence >= 0.7:
            return "high"
        elif confidence >= 0.4:
            return "medium"
        else:
            return "low"

    def _get_market_penalty(self, market_state: str) -> float:
        """根据市场状态返回置信度惩罚项"""
        penalties = {"bull": 0.0, "range": 0.05, "bear": 0.15, "unknown": 0.1}
        return penalties.get(market_state, 0.1)
