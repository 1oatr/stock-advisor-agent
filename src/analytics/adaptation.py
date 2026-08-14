"""analytics/adaptation.py — 标的适配分析

筛选策略最优适配板块、标的类型。
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np


class StockAdaptation:
    """标的适配分析器"""

    def __init__(self):
        pass

    def analyze(self, trade_records: List[dict], stock_info: Optional[Dict[str, dict]] = None) -> Dict:
        """分析各标的的适配度

        Args:
            trade_records: 交易记录列表
            stock_info: {code: {name, sector, market_cap, ...}}

        Returns:
            {code: {total_pnl, trade_count, win_rate, avg_hold_days, adaptation_score}}
        """
        stock_stats = {}
        for t in trade_records:
            code = t.get("code")
            if code not in stock_stats:
                stock_stats[code] = {"trades": 0, "wins": 0, "pnls": [], "hold_days": []}

            stock_stats[code]["trades"] += 1
            stock_stats[code]["pnls"].append(t.get("pnl", 0))
            if t.get("pnl", 0) > 0:
                stock_stats[code]["wins"] += 1

        results = {}
        for code, stats in stock_stats.items():
            pnls = stats["pnls"]
            win_rate = stats["wins"] / max(stats["trades"], 1) * 100
            avg_pnl = np.mean(pnls)
            total_pnl = sum(pnls)
            profit_factor = abs(sum(p for p in pnls if p > 0) / max(abs(sum(p for p in pnls if p < 0)), 1))

            # 综合适配评分
            score = (
                win_rate / 100 * 0.4
                + min(abs(avg_pnl) / 1000, 1) * 0.2
                + min(profit_factor / 3, 1) * 0.2
                + min(stats["trades"] / 20, 1) * 0.2
            )

            results[code] = {
                "total_pnl": round(total_pnl, 2),
                "trade_count": stats["trades"],
                "win_rate": round(win_rate, 2),
                "avg_pnl": round(avg_pnl, 2),
                "profit_factor": round(profit_factor, 2),
                "adaptation_score": round(score, 4),
            }

        # 按适配度排序
        return dict(sorted(results.items(), key=lambda x: x[1]["adaptation_score"], reverse=True))

    def best_stocks(self, analysis: Dict, top_n: int = 5) -> List[Dict]:
        """返回最适配的 N 只股票"""
        return [{"code": k, **v} for k, v in list(analysis.items())[:top_n]]

    def worst_stocks(self, analysis: Dict, top_n: int = 5) -> List[Dict]:
        """返回最不适配的 N 只股票"""
        return [{"code": k, **v} for k, v in list(analysis.items())[-top_n:]]
