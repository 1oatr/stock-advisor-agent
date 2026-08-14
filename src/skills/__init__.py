"""skills — 决策技能模块

通过分析股票前几日的各参数来预测未来走势。
每个技能独立封装一种技术分析方法，接入决策融合层。
"""

from .registry import SkillRegistry, SkillManager
from .candlestick import CandlestickPatterns
from .divergence import RSIDivergence, MACDDivergence
from .breakout import BreakoutAnalysis, SupportResistance
from .momentum import MomentumAnalysis, VolumeAnalysis
from .volatility import VolatilityAnalysis
from .fund_flow import FundFlowSkill
from .volatility_regime import VolatilityRegimeSkill
from .multi_timeframe import MultiTimeframeSkill
# from .news_sentiment import NewsSentiment  # ← 暂时禁用，后续开发再启用

# ---- 自动注册所有技能 ----
_registry_initialized = False


def init_skills():
    """初始化并注册所有技能"""
    global _registry_initialized
    if _registry_initialized:
        return

    skills = [
        CandlestickPatterns(),
        RSIDivergence(),
        MACDDivergence(),
        BreakoutAnalysis(),
        SupportResistance(),
        MomentumAnalysis(),
        VolumeAnalysis(),
        VolatilityAnalysis(),
        FundFlowSkill(),
        VolatilityRegimeSkill(),
        MultiTimeframeSkill(),
        # NewsSentiment(),  # ← 暂时禁用，后续开发再启用
    ]

    for skill in skills:
        SkillRegistry.register(skill)

    _registry_initialized = True


def get_manager() -> SkillManager:
    """获取技能管理器（自动初始化）"""
    init_skills()
    return SkillManager()


__all__ = [
    "SkillRegistry", "SkillManager", "init_skills", "get_manager",
    "CandlestickPatterns", "RSIDivergence", "MACDDivergence",
    "BreakoutAnalysis", "SupportResistance",
    "MomentumAnalysis", "VolumeAnalysis", "VolatilityAnalysis",
    "FundFlowSkill", "VolatilityRegimeSkill", "MultiTimeframeSkill",
    # "NewsSentiment",  # ← 暂时禁用，后续开发再启用
]
