"""advisor/ranker.py — 多股排名器

对多只股票的综合评分进行排序，输出"最推荐买入/卖出"排名。
"""

from typing import List, Dict


class StockRanker:
    """股票排名器

    根据融合决策结果对多只股票打分排序。
    """

    def __init__(self):
        pass

    def rank(self, fused_decisions: List[Dict]) -> List[Dict]:
        """对多只股票的融合决策进行排名

        Args:
            fused_decisions: 每只股票的融合决策列表，每项包含 code, action, confidence,
                            fusion_score, position 等字段

        Returns:
            按推荐优先级排序的列表，买入优先于卖出优先于持有
        """
        if not fused_decisions:
            return []

        # 评分计算：买入为正分、卖出为负分、持有为低分
        scored = []
        for d in fused_decisions:
            code = d.get("code", d.get("stock_code", ""))
            action = d.get("action", "hold")
            fusion_score = d.get("fusion_score", 0.5)
            confidence = d.get("confidence", 0.5)
            position = d.get("position", 0.0)
            reason = d.get("reason", "")

            if action == "buy":
                rank_score = fusion_score * confidence * 100
            elif action == "sell":
                rank_score = -(fusion_score * confidence * 100)
            else:
                rank_score = 10  # 持有放在中间

            scored.append({
                "rank": 0,
                "code": code,
                "name": d.get("name", ""),
                "action": action,
                "fusion_score": fusion_score,
                "confidence": confidence,
                "position": position,
                "rank_score": round(rank_score, 2),
                "reason": reason,
            })

        # 按评分降序排序
        scored.sort(key=lambda x: x["rank_score"], reverse=True)

        # 填充排名
        for i, s in enumerate(scored):
            s["rank"] = i + 1

        return scored

    def top_buy(self, rankings: List[Dict], n: int = 3) -> List[Dict]:
        """返回最推荐的 N 只买入股票"""
        buys = [r for r in rankings if r["action"] == "buy"]
        return buys[:n]

    def top_sell(self, rankings: List[Dict], n: int = 3) -> List[Dict]:
        """返回最建议卖出的 N 只股票"""
        sells = [r for r in rankings if r["action"] == "sell"]
        return sells[:n]

    def format_output(self, rankings: List[Dict]) -> str:
        """格式化输出排名结果（供 CLI 展示）"""
        lines = ["=" * 60, "📊 多股综合推荐排名", "=" * 60]

        if not rankings:
            lines.append("  (无推荐结果)")
            lines.append("=" * 60)
            return "\n".join(lines)

        for r in rankings:
            action_icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}
            icon = action_icon.get(r["action"], "⚪")
            lines.append(
                f"  #{r['rank']} {icon} {r['code']} {r.get('name', '')}\n"
                f"     决策: {r['action']}  |  评分: {r['fusion_score']:.2f}  "
                f"|  置信度: {r['confidence']:.0%}\n"
                f"     建议仓位: {r['position']:.0%}"
            )
            if r.get("reason"):
                lines.append(f"     理由: {r['reason']}")

        lines.append("=" * 60)
        return "\n".join(lines)
