"""skills/base.py — 技能基类

所有决策技能继承此类，实现 evaluate() 方法。
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


# 信号方向常量
SIGNAL_BUY = "buy"
SIGNAL_SELL = "sell"
SIGNAL_HOLD = "hold"
SIGNAL_STRONG_BUY = "strong_buy"
SIGNAL_STRONG_SELL = "strong_sell"

# 置信度等级
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"


@dataclass
class SkillResult:
    """技能分析结果"""
    skill_name: str                    # 技能名称
    signal: str                        # buy / sell / hold / strong_buy / strong_sell
    confidence: float = 0.5            # 置信度 0~1
    strength: float = 0.0              # 信号强度 0~1
    explanation: str = ""              # 分析解释
    patterns_detected: List[str] = field(default_factory=list)   # 检测到的模式
    metadata: Dict = field(default_factory=dict)                  # 额外数据


class BaseSkill:
    """技能基类

    每个技能独立分析数据，输出买卖信号和解释。
    """

    def __init__(self, name: str = "", description: str = "", category: str = ""):
        self.name = name
        self.description = description
        self.category = category

    def evaluate(self, df: pd.DataFrame, code: str = "") -> SkillResult:
        """分析股票近期数据，预测走势

        Args:
            df: 股票近期数据（含OHLCV + 技术指标），至少过去20~60天
            code: 股票代码（可选）

        Returns:
            分析结果
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return f"[{self.category}] {self.name}: {self.description}"
