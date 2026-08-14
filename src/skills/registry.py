"""skills/registry.py — 技能注册表与管理器

统一管理所有技能，支持批量运行和结果汇总。
"""

from typing import Dict, List, Optional, Callable
import pandas as pd
from .base import BaseSkill, SkillResult, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class SkillRegistry:
    """技能注册表"""

    _skills: Dict[str, BaseSkill] = {}

    @classmethod
    def register(cls, skill: BaseSkill):
        """注册技能"""
        cls._skills[skill.name] = skill

    @classmethod
    def get(cls, name: str) -> Optional[BaseSkill]:
        """获取技能"""
        return cls._skills.get(name)

    @classmethod
    def list_by_category(cls) -> Dict[str, List[BaseSkill]]:
        """按类别列出所有技能"""
        categories = {}
        for skill in cls._skills.values():
            if skill.category not in categories:
                categories[skill.category] = []
            categories[skill.category].append(skill)
        return categories

    @classmethod
    def list_all(cls) -> List[BaseSkill]:
        """列出所有技能"""
        return list(cls._skills.values())

    @classmethod
    def count(cls) -> int:
        return len(cls._skills)


class SkillManager:
    """技能管理器

    运行已注册的所有（或指定）技能，并汇总结果。
    """

    def __init__(self):
        self.results: Dict[str, List[SkillResult]] = {}

    def run_all(self, df: pd.DataFrame, code: str = "") -> List[SkillResult]:
        """运行所有已注册技能

        Args:
            df: 股票近期数据
            code: 股票代码

        Returns:
            所有技能的分析结果
        """
        results = []
        for skill in SkillRegistry.list_all():
            try:
                result = skill.evaluate(df, code)
                results.append(result)
            except Exception as e:
                results.append(SkillResult(
                    skill_name=skill.name,
                    signal=SIGNAL_HOLD,
                    confidence=0.0,
                    explanation=f"技能执行异常: {e}",
                ))
        self.results[code] = results
        return results

    def run_by_category(self, df: pd.DataFrame, category: str, code: str = "") -> List[SkillResult]:
        """运行指定类别的技能"""
        results = []
        for skill in SkillRegistry.list_all():
            if skill.category == category:
                try:
                    results.append(skill.evaluate(df, code))
                except Exception as e:
                    results.append(SkillResult(
                        skill_name=skill.name, signal=SIGNAL_HOLD,
                        confidence=0.0, explanation=f"异常: {e}",
                    ))
        return results

    def run_selected(self, df: pd.DataFrame, skill_names: List[str], code: str = "") -> List[SkillResult]:
        """运行指定的技能"""
        results = []
        for name in skill_names:
            skill = SkillRegistry.get(name)
            if skill:
                try:
                    results.append(skill.evaluate(df, code))
                except Exception as e:
                    results.append(SkillResult(
                        skill_name=name, signal=SIGNAL_HOLD,
                        confidence=0.0, explanation=f"异常: {e}",
                    ))
        return results

    def aggregate_signal(self, results: List[SkillResult]) -> Dict:
        """汇总多个技能结果生成综合信号

        加权汇总：买入信号加权、卖出信号加权，胜出者为综合信号。

        Returns:
            {
                "signal": "buy",
                "confidence": 0.72,
                "buy_skills": [...],
                "sell_skills": [...],
                "details": [...]
            }
        """
        if not results:
            return {"signal": SIGNAL_HOLD, "confidence": 0.5, "buy_skills": [], "sell_skills": []}

        total_weight = 0
        buy_score = 0.0
        sell_score = 0.0
        buy_skills = []
        sell_skills = []

        for r in results:
            w = r.confidence * (r.strength + 0.5)  # 权重 = 置信度 × 强度
            total_weight += w

            if r.signal in (SIGNAL_BUY, "strong_buy"):
                buy_score += w * (1.3 if r.signal == "strong_buy" else 1.0)
                buy_skills.append(r.skill_name)
            elif r.signal in (SIGNAL_SELL, "strong_sell"):
                sell_score += w * (1.3 if r.signal == "strong_sell" else 1.0)
                sell_skills.append(r.skill_name)

        if total_weight == 0:
            return {"signal": SIGNAL_HOLD, "confidence": 0.5, "buy_skills": buy_skills, "sell_skills": sell_skills}

        if buy_score > sell_score:
            signal = SIGNAL_BUY
            confidence = buy_score / total_weight
        elif sell_score > buy_score:
            signal = SIGNAL_SELL
            confidence = sell_score / total_weight
        else:
            signal = SIGNAL_HOLD
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": min(confidence, 1.0),
            "buy_skills": buy_skills,
            "sell_skills": sell_skills,
            "details": results,
        }

    def format_results(self, results: List[SkillResult]) -> str:
        """格式化技能分析结果（供展示）"""
        lines = ["\n📊 技能分析结果", "=" * 50]
        if not results:
            lines.append("  (无结果)")
            return "\n".join(lines)

        # 按信号分组
        buys = [r for r in results if r.signal in (SIGNAL_BUY, "strong_buy")]
        sells = [r for r in results if r.signal in (SIGNAL_SELL, "strong_sell")]
        holds = [r for r in results if r.signal == SIGNAL_HOLD]

        if buys:
            lines.append(f"\n🟢 买入信号 ({len(buys)})")
            for r in buys:
                lines.append(f"  ✅ {r.skill_name} (置信度: {r.confidence:.0%})")
                lines.append(f"     {r.explanation}")

        if sells:
            lines.append(f"\n🔴 卖出信号 ({len(sells)})")
            for r in sells:
                lines.append(f"  ❌ {r.skill_name} (置信度: {r.confidence:.0%})")
                lines.append(f"     {r.explanation}")

        if holds:
            lines.append(f"\n⚪ 持有/无信号 ({len(holds)})")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)
