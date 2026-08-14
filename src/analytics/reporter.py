"""analytics/reporter.py — 定期报告生成

日 / 周 / 月 量化交易总结、风险评估报告。
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd


class ReportGenerator:
    """报告生成器"""

    def __init__(self, log_dir: str = "data/reports"):
        self.log_dir = log_dir
        import os
        os.makedirs(log_dir, exist_ok=True)

    def generate_daily(self, metrics: dict, today_trades: List[dict],
                       positions: dict, risk_status: dict) -> str:
        """生成日报"""
        now = datetime.now()
        lines = [
            f"={'='*50}",
            f"  每日量化交易报告",
            f"  {now.strftime('%Y-%m-%d %H:%M')}",
            f"={'='*50}",
            "",
            "【当日概览】",
            f"  总资产: {metrics.get('total_equity', 0):,.2f}",
            f"  持仓市值: {metrics.get('market_value', 0):,.2f}",
            f"  现金: {metrics.get('cash', 0):,.2f}",
            f"  当日盈亏: {metrics.get('daily_pnl', 0):+,.2f}",
            f"  当日交易: {len(today_trades)} 笔",
            "",
            "【持仓明细】",
        ]
        for code, pos in positions.items():
            lines.append(f"  {code}: {pos.get('volume', 0)}股 "
                         f"成本{pos.get('avg_cost', 0):.2f} "
                         f"现价{pos.get('current_price', 0):.2f} "
                         f"盈亏{pos.get('pnl_pct', 0):+.2f}%")

        lines.extend([
            "",
            "【风控状态】",
            f"  当前回撤: {risk_status.get('current_drawdown', 0):.2f}%",
            f"  风控等级: {risk_status.get('level', 'normal')}",
            "",
            "【信号记录】",
        ])
        for t in today_trades[-10:]:  # 最近10笔
            lines.append(f"  {t.get('timestamp', '')} {t.get('code','')} "
                         f"{t.get('direction','')} {t.get('price',0):.2f}")

        report = "\n".join(lines)
        self._save_report("daily", report)
        return report

    def generate_weekly(self, metrics: dict, weekly_trades: List[dict],
                        attribution: dict, adaptation: dict) -> str:
        """生成周报"""
        lines = [
            f"={'='*50}",
            f"  周度量化交易总结",
            f"  {datetime.now().strftime('%Y-%m-%d')}",
            f"={'='*50}",
            "",
            "【本周绩效】",
            f"  总收益率: {metrics.get('total_return', 0):.2f}%",
            f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%",
            f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}",
            f"  胜率: {metrics.get('win_rate', 0):.2f}%",
            f"  交易次数: {len(weekly_trades)}",
            "",
            "【盈亏归因】",
        ]
        if attribution:
            for regime, data in attribution.items():
                lines.append(f"  {regime}: 收益{data.get('return', 0):.2f}% "
                             f"胜率{data.get('win_rate', 0):.1f}% "
                             f"{data.get('trade_count', 0)}笔")

        lines.extend([
            "",
            "【标的适配 Top3】",
        ])
        if adaptation:
            for i, (code, info) in enumerate(list(adaptation.items())[:3], 1):
                lines.append(f"  {i}. {code}: 评分{info.get('adaptation_score', 0):.3f} "
                             f"胜率{info.get('win_rate', 0):.1f}%")

        report = "\n".join(lines)
        self._save_report("weekly", report)
        return report

    def generate_monthly(self, metrics: dict, monthly_trades: List[dict],
                         attribution: dict, adaptation: dict,
                         optimization: dict) -> str:
        """生成月报"""
        lines = [
            f"={'='*60}",
            f"  月度量化交易总结报告",
            f"  {datetime.now().strftime('%Y-%m')}",
            f"={'='*60}",
            "",
            "【月度核心指标】",
            f"  总收益率: {metrics.get('total_return', 0):.2f}%",
            f"  年化收益: {metrics.get('annual_return', 0):.2f}%",
            f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%",
            f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}",
            f"  索提诺比率: {metrics.get('sortino_ratio', 0):.2f}",
            f"  卡玛比率: {metrics.get('calmar_ratio', 0):.2f}",
            f"  胜率: {metrics.get('win_rate', 0):.1f}%",
            f"  盈亏比: {metrics.get('profit_loss_ratio', 0):.2f}",
            f"  总交易: {metrics.get('total_trades', 0)} 笔",
            f"  日均波动: {metrics.get('daily_volatility', 0):.2f}%",
            "",
            "【风险评估】",
            f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%",
            f"  回撤天数: {metrics.get('drawdown_days', 0)} 天",
            f"  收益稳定性: {metrics.get('profit_stability', 0):.4f}",
        ]

        if optimization:
            lines.extend([
                "",
                "【参数优化】",
                f"  搜索次数: {optimization.get('searched', 0)}",
                f"  最优参数: {optimization.get('best_params', {})}",
                f"  最优评分: {optimization.get('best_score', 0):.4f}",
            ])

        report = "\n".join(lines)
        self._save_report("monthly", report)
        return report

    def _save_report(self, report_type: str, content: str):
        """保存报告到文件"""
        now = datetime.now()
        ext = {"daily": now.strftime("%Y%m%d"), "weekly": now.strftime("%Y_W%W"),
               "monthly": now.strftime("%Y%m")}
        filename = f"{report_type}_{ext.get(report_type, 'unknown')}.txt"
        path = f"{self.log_dir}/{filename}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
