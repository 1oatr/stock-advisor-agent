"""backtest/metrics.py — 核心绩效指标计算

收益率、最大回撤、夏普比率、胜率、盈亏比、交易次数、持仓周期、年化收益。
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np


class PerformanceMetrics:
    """绩效指标计算器"""

    @staticmethod
    def calculate_all(equity_curve: pd.DataFrame, trades: List[dict], risk_free_rate: float = 0.025) -> dict:
        """计算所有核心绩效指标

        Args:
            equity_curve: 净值曲线 [{date, total, cash, market_value}]
            trades: 交易记录列表
            risk_free_rate: 无风险利率

        Returns:
            完整的绩效指标字典
        """
        metrics = {}

        # ---- 收益指标 ----
        metrics.update(PerformanceMetrics._return_metrics(equity_curve))

        # ---- 风险指标 ----
        metrics.update(PerformanceMetrics._risk_metrics(equity_curve, risk_free_rate))

        # ---- 交易统计 ----
        metrics.update(PerformanceMetrics._trade_metrics(trades))

        # ---- 稳定性指标 ----
        metrics.update(PerformanceMetrics._stability_metrics(equity_curve))

        return metrics

    @staticmethod
    def _return_metrics(equity: pd.DataFrame) -> dict:
        """收益相关指标"""
        if equity.empty:
            return {}

        initial = equity["total"].iloc[0]
        final = equity["total"].iloc[-1]
        days = len(equity)
        years = days / 252

        total_return = (final / initial - 1) * 100
        annual_return = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

        # 每日收益率
        equity["daily_ret"] = equity["total"].pct_change()

        return {
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "total_days": days,
            "trading_years": round(years, 2),
        }

    @staticmethod
    def _risk_metrics(equity: pd.DataFrame, rfr: float = 0.025) -> dict:
        """风险相关指标"""
        if equity.empty or len(equity) < 2:
            return {}

        initial = equity["total"].iloc[0]
        rolling_max = equity["total"].cummax()
        drawdown = (equity["total"] - rolling_max) / rolling_max * 100

        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        drawdown_days = (drawdown < 0).sum()

        equity["daily_ret"] = equity["total"].pct_change()
        daily_std = equity["daily_ret"].std()

        # 夏普比率
        excess_returns = equity["daily_ret"] - rfr / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / (daily_std + 1e-8)

        # 卡玛比率（收益/最大回撤）
        total_return = (equity["total"].iloc[-1] / initial - 1) * 100
        calmar = abs(total_return / max_drawdown) if max_drawdown != 0 else float("inf")

        # 索提诺比率（只考虑下行波动）
        downside_ret = equity["daily_ret"][equity["daily_ret"] < 0]
        downside_std = downside_ret.std()
        sortino = np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-8)

        return {
            "max_drawdown": round(max_drawdown, 2),
            "avg_drawdown": round(avg_drawdown, 2),
            "drawdown_days": drawdown_days,
            "sharpe_ratio": round(sharpe, 2),
            "calmar_ratio": round(calmar, 2),
            "sortino_ratio": round(sortino, 2),
            "daily_volatility": round(daily_std * 100, 2),
        }

    @staticmethod
    def _trade_metrics(trades: List[dict]) -> dict:
        """交易统计指标"""
        if not trades:
            return {"total_trades": 0, "win_rate": 0, "profit_loss_ratio": 0}

        total = len(trades)
        # 只算卖出才有盈亏
        sell_trades = [t for t in trades if t.get("direction") == "sell" and t.get("pnl", 0) != 0]

        if not sell_trades:
            return {"total_trades": total, "win_rate": 0, "profit_loss_ratio": 0}

        wins = [t for t in sell_trades if t["pnl"] > 0]
        losses = [t for t in sell_trades if t["pnl"] < 0]

        win_rate = len(wins) / len(sell_trades) * 100
        avg_profit = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1

        return {
            "total_trades": total,
            "sell_trades": len(sell_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "profit_loss_ratio": round(avg_profit / (avg_loss + 1e-8), 2),
            "avg_profit_per_trade": round(avg_profit, 2),
            "avg_loss_per_trade": round(avg_loss, 2),
        }

    @staticmethod
    def _stability_metrics(equity: pd.DataFrame) -> dict:
        """稳定性指标"""
        if equity.empty or "daily_ret" not in equity.columns:
            return {}

        # 收益稳定性（回归R²）
        equity["cum_ret"] = (1 + equity["daily_ret"].fillna(0)).cumprod()
        x = np.arange(len(equity))
        y = equity["cum_ret"].values
        if len(x) > 1:
            corr = np.corrcoef(x, y)[0, 1]
            stability = corr ** 2
        else:
            stability = 0

        return {
            "profit_stability": round(stability, 4),
        }
