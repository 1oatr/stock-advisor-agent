"""advisor/kelly.py — 凯利公式仓位计算

f* = (b × p - q) / b
    b: 赔率（盈利/亏损比）
    p: 获胜概率
    q: 失败概率 = 1-p

实际仓位 = max(0, f*) × k × market_factor
    k: 凯利系数（满凯利太激进，实战用半凯利/1/3凯利）

边界:
    - p 夹在 [0.05, 0.95]，避免极端值
    - f* ≤ 0 → 不下注
    - 最终仓位不超过 max_position_ratio (默认 20%)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class KellyParams:
    """凯利公式参数"""
    b: float = 4.0                # 赔率（默认 止盈20% / 止损5% = 4.0）
    p: float = 0.5                # 获胜概率
    k_high: float = 0.50          # 高共识 → 半凯利
    k_medium: float = 0.35        # 中共识 → ≈1/3凯利
    k_low: float = 0.20           # 低共识 → 保守
    max_position: float = 0.20    # 单标的最大仓位上限
    p_min: float = 0.05           # p 下限
    p_max: float = 0.95           # p 上限


class KellyPosition:
    """凯利公式仓位计算器

    用法:
        kelly = KellyPosition()
        position = kelly.calculate(
            confidence=0.55,           # p — 融合置信度
            consensus="medium",        # 三路共识度
            market_state="range",      # 大盘状态
            odds=4.0,                  # b — 赔率（可选，默认 4.0）
        )
    """

    def __init__(self, params: Optional[KellyParams] = None):
        self.params = params or KellyParams()

    # ========================================================================
    # 公开 API
    # ========================================================================

    def calculate(
        self,
        confidence: float,
        consensus: str = "medium",
        market_state: str = "unknown",
        odds: Optional[float] = None,
    ) -> float:
        """计算凯利最优仓位

        Args:
            confidence: 融合置信度 0~1，作为获胜概率 p
            consensus: 三路共识度 "high" / "medium" / "low"
            market_state: 大盘状态 "bull" / "bear" / "range"
            odds: 赔率 b，默认取 params.b (4.0)

        Returns:
            position: 建议仓位比例 0.0 ~ max_position
        """
        b = odds if odds is not None else self.params.b
        p = self._clamp_p(confidence)

        # 核心公式
        q = 1.0 - p
        if b <= 0:
            return 0.0

        f_star = (b * p - q) / b

        # 凯利说不划算 → 不下注
        if f_star <= 0:
            return 0.0

        # 凯利系数 × 大盘系数
        k = self._consensus_k(consensus)
        m = self._market_factor(market_state)

        position = f_star * k * m
        return round(min(position, self.params.max_position), 4)

    def diagnose(
        self,
        confidence: float,
        consensus: str = "medium",
        market_state: str = "unknown",
        odds: Optional[float] = None,
    ) -> dict:
        """调试用：返回完整计算过程"""
        b = odds if odds is not None else self.params.b
        p = self._clamp_p(confidence)
        q = 1.0 - p
        f_star = (b * p - q) / b if b > 0 else 0.0
        k = self._consensus_k(consensus)
        m = self._market_factor(market_state)
        position = max(0, f_star) * k * m
        position = min(position, self.params.max_position)

        return {
            "b": b,
            "p": p,
            "q": q,
            "f_star": round(f_star, 4),
            "consensus": consensus,
            "k": k,
            "market_state": market_state,
            "market_factor": m,
            "position_raw": round(max(0, f_star) * k * m, 4),
            "position_final": round(position, 4),
            "max_cap": self.params.max_position,
            "capped": position >= self.params.max_position,
        }

    # ========================================================================
    # 内部
    # ========================================================================

    def _clamp_p(self, p: float) -> float:
        return max(self.params.p_min, min(self.params.p_max, p))

    def _consensus_k(self, consensus: str) -> float:
        """共识度 → 凯利系数"""
        return {
            "high": self.params.k_high,
            "medium": self.params.k_medium,
            "low": self.params.k_low,
        }.get(consensus, self.params.k_medium)

    def _market_factor(self, market_state: str) -> float:
        """大盘 → 仓位折扣"""
        return {
            "bull": 1.0,
            "range": 0.8,
            "bear": 0.5,
        }.get(market_state, 0.8)  # 未知默认按震荡


# ----------------------------------------------------------------------
# 单例
# ----------------------------------------------------------------------
_kelly: Optional[KellyPosition] = None


def get_kelly() -> KellyPosition:
    global _kelly
    if _kelly is None:
        _kelly = KellyPosition()
    return _kelly
