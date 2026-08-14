"""backtest/costs.py — 交易成本模拟

手续费、印花税、滑点计算。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CostConfig:
    """交易成本配置"""
    commission_rate: float = 0.00025      # 佣金 万分之2.5
    min_commission: float = 5.0           # 最低佣金 5元
    stamp_tax_rate: float = 0.001         # 印花税 千分之一（仅卖出）
    transfer_fee_rate: float = 0.00002    # 过户费 十万分之二
    slippage_rate: float = 0.001          # 滑点 0.1%
    is_shenzhen: bool = True               # 深市/沪市（过户费差异）


class CostCalculator:
    """交易成本计算器"""

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def calculate_buy_cost(self, price: float, volume: int) -> dict:
        """买入费用计算"""
        amount = price * volume
        commission = max(amount * self.config.commission_rate, self.config.min_commission)
        transfer_fee = amount * self.config.transfer_fee_rate
        total = commission + transfer_fee
        return {
            "amount": amount,
            "commission": round(commission, 3),
            "transfer_fee": round(transfer_fee, 3),
            "total_cost": round(total, 3),
            "avg_cost_per_share": round(total / volume, 4) if volume > 0 else 0,
        }

    def calculate_sell_cost(self, price: float, volume: int) -> dict:
        """卖出费用计算"""
        amount = price * volume
        commission = max(amount * self.config.commission_rate, self.config.min_commission)
        stamp_tax = amount * self.config.stamp_tax_rate
        transfer_fee = amount * self.config.transfer_fee_rate
        total = commission + stamp_tax + transfer_fee
        return {
            "amount": amount,
            "commission": round(commission, 3),
            "stamp_tax": round(stamp_tax, 3),
            "transfer_fee": round(transfer_fee, 3),
            "total_cost": round(total, 3),
            "avg_cost_per_share": round(total / volume, 4) if volume > 0 else 0,
        }

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """应用滑点"""
        if is_buy:
            return price * (1 + self.config.slippage_rate)
        return price * (1 - self.config.slippage_rate)
