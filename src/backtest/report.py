"""backtest/report.py — 可视化报告

自动生成收益曲线、回撤曲线、交易分布、盈亏明细图表。
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime


class BacktestReport:
    """回测报告生成器"""

    def __init__(self, equity_curve: pd.DataFrame, trades: List[dict], metrics: dict):
        self.equity = equity_curve
        self.trades = trades
        self.metrics = metrics

    def generate_report(self, output_path: str = ""):
        """生成完整回测报告（多图表组合）"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 收益曲线
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1)

        # 回撤曲线
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(ax2)

        # 月度收益热力图
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_monthly_returns(ax3)

        # 交易分布
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_trade_distribution(ax4)

        # 关键指标卡片
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_metrics_cards(ax5)

        plt.suptitle(f"量化回测报告 — {datetime.now().strftime('%Y-%m-%d')}", fontsize=14, y=0.98)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _plot_equity_curve(self, ax):
        """绘制收益曲线"""
        if self.equity.empty:
            return
        dates = pd.to_datetime(self.equity["date"]) if "date" in self.equity else range(len(self.equity))
        ax.plot(dates, self.equity["total"], label="资产净值", color="#2196F3", linewidth=2)
        ax.fill_between(dates, self.equity["total"], alpha=0.1, color="#2196F3")
        ax.axhline(y=self.equity["total"].iloc[0], color="gray", linestyle="--", alpha=0.5)
        ax.set_title("收益曲线")
        ax.set_ylabel("资产净值 (元)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_drawdown(self, ax):
        """绘制回撤曲线"""
        if self.equity.empty:
            return
        rolling_max = self.equity["total"].cummax()
        drawdown = (self.equity["total"] - rolling_max) / rolling_max * 100
        dates = pd.to_datetime(self.equity["date"]) if "date" in self.equity else range(len(self.equity))
        ax.fill_between(dates, drawdown, 0, color="#f44336", alpha=0.5)
        ax.set_title(f"回撤曲线 (最大: {self.metrics.get('max_drawdown', 0):.1f}%)")
        ax.set_ylabel("回撤 (%)")
        ax.grid(True, alpha=0.3)

    def _plot_monthly_returns(self, ax):
        """绘制月度收益热力图"""
        # TODO: 月度收益率矩阵
        ax.text(0.5, 0.5, "月度收益热力图\n(待实现)", ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("月度收益")

    def _plot_trade_distribution(self, ax):
        """绘制交易盈亏分布"""
        sell_trades = [t for t in self.trades if t.get("pnl", 0) != 0]
        if not sell_trades:
            ax.text(0.5, 0.5, "暂无交易数据", ha="center", va="center", fontsize=12, color="gray")
            ax.set_title("盈亏分布")
            return

        pnls = [t["pnl"] for t in sell_trades]
        colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(f"盈亏分布 (胜率: {self.metrics.get('win_rate', 0):.1f}%)")
        ax.set_xlabel("交易序号")
        ax.set_ylabel("盈亏 (元)")

    def _plot_metrics_cards(self, ax):
        """绘制关键指标卡片"""
        ax.axis("off")
        cards = [
            ("总收益率", f"{self.metrics.get('total_return', 0):.1f}%", "#4CAF50"),
            ("年化收益", f"{self.metrics.get('annual_return', 0):.1f}%", "#2196F3"),
            ("最大回撤", f"{self.metrics.get('max_drawdown', 0):.1f}%", "#f44336"),
            ("夏普比率", f"{self.metrics.get('sharpe_ratio', 0):.2f}", "#FF9800"),
            ("胜率", f"{self.metrics.get('win_rate', 0):.1f}%", "#9C27B0"),
            ("盈亏比", f"{self.metrics.get('profit_loss_ratio', 0):.2f}", "#00BCD4"),
            ("总交易", f"{self.metrics.get('total_trades', 0)}", "#607D8B"),
            ("卡玛比率", f"{self.metrics.get('calmar_ratio', 0):.2f}", "#795548"),
        ]
        for i, (name, value, color) in enumerate(cards):
            row = i // 4
            col = i % 4
            x = col * 0.24 + 0.05
            y = 0.55 - row * 0.4
            ax.text(x + 0.06, y + 0.08, name, fontsize=9, color="gray", ha="center")
            ax.text(x + 0.06, y - 0.02, value, fontsize=18, color=color, fontweight="bold", ha="center")
