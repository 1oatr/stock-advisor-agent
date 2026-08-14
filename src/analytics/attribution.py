"""analytics/attribution.py — 策略盈亏归因

区分趋势行情、震荡行情下的策略表现，定位亏损场景。
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np


class StrategyAttribution:
    """策略盈亏归因分析"""

    def __init__(self, equity_curve: pd.DataFrame, trades: List[dict], index_data: Optional[pd.DataFrame] = None):
        self.equity = equity_curve
        self.trades = trades
        self.index = index_data

    def analyze_by_market_regime(self) -> Dict:
        """按市场状态（趋势/震荡）分析策略表现"""
        regimes = self._classify_market_regimes()

        results = {}
        for regime_name, periods in regimes.items():
            period_equity = self.equity[self.equity["date"].isin(periods)]
            period_trades = [t for t in self.trades if t.get("date") in periods]

            if period_equity.empty:
                continue

            ret = (period_equity["total"].iloc[-1] / period_equity["total"].iloc[0] - 1) * 100
            wins = [t for t in period_trades if t.get("pnl", 0) > 0]
            losses = [t for t in period_trades if t.get("pnl", 0) < 0]

            results[regime_name] = {
                "return": round(ret, 2),
                "trade_count": len(period_trades),
                "win_rate": round(len(wins) / max(len(period_trades), 1) * 100, 2),
                "avg_win": round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
                "avg_loss": round(abs(np.mean([t["pnl"] for t in losses])), 2) if losses else 0,
            }

        return results

    def analyze_by_sector(self, code_to_sector: Dict[str, str]) -> Dict:
        """按行业板块归因"""
        sector_pnls = {}
        for t in self.trades:
            code = t.get("code")
            if code and code in code_to_sector:
                sector = code_to_sector[code]
                if sector not in sector_pnls:
                    sector_pnls[sector] = {"trades": 0, "pnl": 0, "wins": 0}
                sector_pnls[sector]["trades"] += 1
                sector_pnls[sector]["pnl"] += t.get("pnl", 0)
                if t.get("pnl", 0) > 0:
                    sector_pnls[sector]["wins"] += 1
        return sector_pnls

    def analyze_loss_scenarios(self) -> List[Dict]:
        """分析亏损场景，定位原因"""
        losses = [t for t in self.trades if t.get("pnl", 0) < 0]
        if not losses:
            return []

        analysis = {
            "total_losses": len(losses),
            "total_loss_amount": round(sum(t["pnl"] for t in losses), 2),
            "avg_loss": round(np.mean([t["pnl"] for t in losses]), 2),
            "max_loss": round(min(t["pnl"] for t in losses), 2),
            "top_loss_stocks": self._top_loss_stocks(losses),
        }
        return analysis

    def _classify_market_regimes(self) -> Dict[str, list]:
        """划分市场状态"""
        # TODO: 使用指数数据判断趋势/震荡/下跌区间
        return {"trend_up": [], "trend_down": [], "range_bound": []}

    def _top_loss_stocks(self, losses: List[dict], top_n: int = 5) -> List[dict]:
        """亏损最多的股票"""
        stock_losses = {}
        for t in losses:
            code = t.get("code")
            stock_losses[code] = stock_losses.get(code, 0) + t.get("pnl", 0)
        sorted_stocks = sorted(stock_losses.items(), key=lambda x: x[1])[:top_n]
        return [{"code": code, "total_loss": round(loss, 2)} for code, loss in sorted_stocks]
