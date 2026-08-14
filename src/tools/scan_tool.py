"""tools/scan_tool.py — 全市场扫描工具"""

from src.scanner.scanner import scan_hot_stocks
from src.scanner.market_watch import MarketWatch
from src.data.fetcher import DataFetcher


def scan_market(top_n: int = 15) -> dict:
    """扫描全市场活跃股票

    Returns:
        {"market_state": "bear", "suggested_position": 20, "hot_stocks": [...]}
    """
    fetcher = DataFetcher()
    market = MarketWatch(fetcher=fetcher)
    state = market.update()

    results = scan_hot_stocks(top_n=top_n, fetcher=fetcher)

    # 精简输出
    stocks = []
    for r in results:
        stocks.append({
            "rank": len(stocks) + 1,
            "code": r["code"],
            "name": r.get("name", ""),
            "score": r["score"],
            "close": r.get("last_close", 0),
            "momentum": r.get("momentum", 0),
            "volume_ratio": r.get("volume_ratio", 0),
        })

    return {
        "market_state": state.get("state", "unknown"),
        "index_return_20d": round(state.get("ret_20d", 0), 2),
        "suggested_position_pct": round(market.suggest_position_level() * 100),
        "scanned_count": len(stocks),
        "hot_stocks": stocks,
    }
