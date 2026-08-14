"""tools/backtest_tool.py — 回测工具"""

from src.data.fetcher import DataFetcher
from src.data.indicators import add_all_indicators
from src.backtest.engine import BacktestEngine
from src.strategy.templates import get_strategy


def run_backtest(codes: list, strategy: str = "trend_following",
                 start: str = "2024-01-01", end: str = "2025-06-30",
                 initial_cash: int = 1_000_000) -> dict:
    """运行策略回测

    Returns:
        {"strategy": "trend_following", "metrics": {...}, "stock_count": N}
    """
    fetcher = DataFetcher()
    data = {}

    for code in codes:
        df = fetcher.daily_bars(code, start=start, end=end)
        if df.empty:
            continue
        df = add_all_indicators(df)
        data[code] = df.set_index("date")

    if not data:
        return {"error": "无回测数据", "codes": codes}

    # 获取策略
    try:
        strat = get_strategy(strategy)
    except ValueError:
        return {"error": f"未知策略: {strategy}"}

    def signal_fn(date, positions, cash, day_data):
        return strat.next(date, day_data, positions, cash)

    engine = BacktestEngine(initial_cash=initial_cash)
    metrics = engine.run(data, signal_fn)

    return {
        "strategy": strategy,
        "stock_count": len(data),
        "date_range": f"{start} ~ {end}",
        "metrics": {
            "total_return_pct": metrics.get("total_return", 0),
            "annual_return_pct": metrics.get("annual_return", 0),
            "max_drawdown_pct": metrics.get("max_drawdown", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "win_rate_pct": metrics.get("win_rate", 0),
            "total_trades": metrics.get("total_trades", 0),
            "final_equity": metrics.get("final_cash", initial_cash),
        },
    }
