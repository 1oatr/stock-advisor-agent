"""webui/services.py — 业务适配层

复用现有 src/ 工具函数，只做 JSON 友好化转换。WebUI 不重写任何分析逻辑。

涵盖: 搜索 / K线 / 实时行情 / 分析 / 技能 / RL / 预测 / 历史 / 自选股 / 大盘
"""

import logging
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# 简单进程内缓存（akshare 接口限流保护）
# ============================================================================


class TTLCache:
    """线程安全的 TTL 缓存"""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._data = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self._ttl:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key, value):
        with self._lock:
            self._data[key] = (time.time(), value)


# 实时快照全表缓存 15s（防东财限流）；个股 info/fund_flow 缓存更长
_spot_cache = TTLCache(15)
_info_cache = TTLCache(24 * 3600)
_fund_cache = TTLCache(600)
_kline_cache = TTLCache(300)
_fail_cache = TTLCache(60)  # 东财失败标记（60s 内不重复重试）


def _get_fetcher():
    from src.data.fetcher import DataFetcher
    return DataFetcher()


def clear_stock_cache(code: str):
    """清除某股票的分析/技能/RL 结果缓存（训练完成后调用）"""
    for cache in (_fund_cache, _spot_cache, _info_cache, _fin_cache):
        for key in list(cache._data.keys()):
            if isinstance(key, str) and key.endswith(f":{code}"):
                with cache._lock:
                    cache._data.pop(key, None)
    # 清除失败标记
    for key in list(_fail_cache._data.keys()):
        if isinstance(key, str) and key.endswith(f":{code}"):
            with _fail_cache._lock:
                _fail_cache._data.pop(key, None)


def _get_name_resolver():
    from src.data.name_resolver import get_name_resolver
    return get_name_resolver()


def _get_interested():
    from src.memory import get_interested_stocks
    return get_interested_stocks()


# ============================================================================
# 搜索
# ============================================================================

def search_stocks(keyword: str, limit: int = 10) -> dict:
    """搜索股票：名称模糊 + 6 位代码前缀匹配"""
    keyword = keyword.strip()
    if not keyword:
        return {"results": []}

    resolver = _get_name_resolver()
    results = []

    # 名称模糊搜索
    try:
        for code, name in resolver.search(keyword, limit):
            results.append({"code": code, "name": name})
    except Exception as e:
        logger.warning(f"名称搜索失败: {e}")

    # 代码前缀匹配（6 位数字）
    if keyword.isdigit() and len(keyword) <= 6:
        added = {r["code"] for r in results}
        prefix_hits = [
            c for c in resolver._code_to_name
            if c.startswith(keyword) and c not in added
        ][:limit]
        for code in prefix_hits:
            results.append({"code": code, "name": resolver._code_to_name[code]})

    # 去重
    seen, dedup = set(), []
    for r in results:
        if r["code"] not in seen:
            seen.add(r["code"])
            dedup.append(r)
        if len(dedup) >= limit:
            break

    return {"results": dedup}


def resolve_name(code: str) -> str:
    """代码 → 名称（市场感知）"""
    from .market_catalog import parse_code, resolve_market_name
    market, symbol = parse_code(code)
    if market == "a":
        return _get_name_resolver().resolve_name(symbol) or symbol
    return resolve_market_name(market, symbol)


# ============================================================================
# K 线
# ============================================================================

def _normalize_market_df(df: pd.DataFrame, has_amount: bool,
                         has_turnover: bool = False) -> pd.DataFrame:
    """把港股/美股日线统一为 A 股格式: date/open/high/low/close/volume[/amount/turnover_rate]"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if has_amount and "amount" not in df.columns:
        df["amount"] = 0.0
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    if has_turnover and "turnover_rate" not in df.columns:
        df["turnover_rate"] = None
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = None
    return df.sort_values("date").reset_index(drop=True)


# akshare 的新浪日线依赖 py_mini_racer JS 引擎，该库在 Windows 上**非线程安全**，
# 并发调用会偶发段错误崩溃。新浪源的所有数据拉取用全局锁串行化。
# 东财源（requests 实现）不需要锁，可与新浪源并行。
_fetch_lock = threading.Lock()


def _fetch_sina_daily(market: str, symbol: str, start: str = "", end: str = "") -> pd.DataFrame:
    """新浪源日线（有锁，防 MiniRacer 崩溃）"""
    with _fetch_lock:
        try:
            if market == "a":
                fetcher = _get_fetcher()
                df = fetcher.daily_bars(symbol, start=start, end=end)
                if "turnover_rate" not in df.columns and "turnover" in df.columns:
                    df["turnover_rate"] = df["turnover"]
                return df
            if market == "hk":
                import akshare as ak
                df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
                return _normalize_market_df(df, has_amount=True)
            if market == "us":
                import akshare as ak
                df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
                return _normalize_market_df(df, has_amount=False)
        except Exception as e:
            logger.warning(f"新浪源 {market}:{symbol} 失败: {e}")
        return pd.DataFrame()


def _fetch_em_daily(market: str, symbol: str, lmt: int = 600) -> pd.DataFrame:
    """东财源日线（requests，无锁，服务端过滤，快）

    secid: A股 1.600519 / 港股 116.00700 / 美股 105.AAPL
    klines 格式: "日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率"
    """
    try:
        import requests
        if market == "a":
            prefix = "1." if symbol.startswith(("6", "9", "5")) else "0."
            secid = prefix + symbol
        elif market == "hk":
            secid = "116." + symbol.zfill(5)
        else:
            secid = "105." + symbol.upper()
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": secid, "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101", "fqt": "1", "beg": "0", "end": "20500101", "lmt": str(lmt),
        }
        r = requests.get(url, params=params, timeout=2,
                         headers={"User-Agent": "Mozilla/5.0"},
                         proxies={"http": None, "https": None})
        data = r.json().get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return pd.DataFrame()
        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) < 7:
                continue
            row = {
                "date": parts[0], "open": float(parts[1]), "close": float(parts[2]),
                "high": float(parts[3]), "low": float(parts[4]),
                "volume": float(parts[5]), "amount": float(parts[6]),
            }
            if len(parts) >= 11:
                row["turnover_rate"] = float(parts[10])  # 换手率
            rows.append(row)
        return _normalize_market_df(pd.DataFrame(rows), has_amount=True, has_turnover=True)
    except Exception as e:
        logger.debug(f"东财源 {market}:{symbol} 失败: {e}")
        return pd.DataFrame()


def fetch_daily_df(market: str, symbol: str, start: str = "", end: str = "") -> pd.DataFrame:
    """多数据源并行获取日线，谁先成功用谁（不等最慢的）

    东财源（requests，快，但可能网络不可用）与新浪源（akshare，稳但慢）
    并行发起，**任一成功即返回**（不等另一个），都失败返回空。

    优先东财（快、含换手率），东财不可用时新浪兜底。
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    result_box = {}
    done = threading.Event()
    box_lock = threading.Lock()

    def _try(fn, label):
        try:
            df = fn()
            if not df.empty:
                with box_lock:
                    if label not in result_box:
                        result_box[label] = df
                done.set()
        except Exception as e:
            logger.debug(f"{label} 拉取失败: {e}")

    ex = ThreadPoolExecutor(max_workers=2)
    ex.submit(_try, lambda: _fetch_em_daily(market, symbol), "em")
    ex.submit(_try, lambda: _fetch_sina_daily(market, symbol, start, end), "sina")
    # 任一成功即返回；最迟等 12s（新浪源上限）
    import time as _t
    deadline = _t.time() + 12
    while _t.time() < deadline and not done.is_set():
        _t.sleep(0.05)
    ex.shutdown(wait=False)  # 后台线程自生自灭，不阻塞返回

    # 优先东财，否则新浪
    if "em" in result_box:
        return result_box["em"]
    if "sina" in result_box:
        return result_box["sina"]
    return pd.DataFrame()


def get_kline(code: str, days: int = 500) -> dict:
    """日线 K 线数据（ECharts 格式，多市场支持）

    数据源: 新浪日线（A股/港股/美股）。
    返回 candles 顺序为 ECharts 要求的 [open, close, low, high]。
    """
    from .market_catalog import parse_code
    code = code.strip()
    market, symbol = parse_code(code)
    cache_key = f"kline:{code}:{days}"
    cached = _kline_cache.get(cache_key)
    if cached is not None:
        return cached

    df = fetch_daily_df(market, symbol)
    if df.empty:
        return {"error": f"未获取到 {code} 数据", "code": code, "market": market}

    df = df.tail(days).reset_index(drop=True)

    last = df.iloc[-1]
    first = df.iloc[0]
    period_return = round((last["close"] / first["close"] - 1) * 100, 2)

    result = {
        "code": code,
        "market": market,
        "name": resolve_name(code),
        "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "candles": df[["open", "close", "low", "high"]].astype(float).values.tolist(),
        "volumes": df["volume"].astype(float).tolist(),
        "volume_dirs": np.where(df["close"] >= df["open"], 1, -1).tolist(),
        "close": df["close"].astype(float).tolist(),
        "turnover_rates": _to_nullable_list(df, "turnover_rate"),
        "ma5": _rolling_ma(df["close"], 5),
        "ma20": _rolling_ma(df["close"], 20),
        "ma60": _rolling_ma(df["close"], 60),
        "meta": {
            "last_close": round(float(last["close"]), 2),
            "period_return": period_return,
            "data_days": len(df),
            "data_range": f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}",
        },
    }
    _kline_cache.set(cache_key, result)
    return result


def _rolling_ma(series: pd.Series, n: int) -> list:
    """滚动均线，前 n-1 个填 None"""
    return _to_nullable_list(series.rolling(n).mean(), "ma")


def _to_nullable_list(df_or_series, col) -> list:
    """pandas NaN → Python None（JSON null）"""
    s = df_or_series[col] if isinstance(df_or_series, pd.DataFrame) else df_or_series
    return [None if pd.isna(v) else float(v) for v in s]


# ============================================================================
# 实时行情（东财，带缓存与降级）
# ============================================================================

def get_quote(code: str) -> dict:
    """实时行情：现价/涨跌/市盈率/换手率/量比/市值/外盘内盘/主力净流入

    数据源（东财，需禁代理）:
      - stock_zh_a_spot_em  全市场快照 → 现价/PE/换手/量比/市值
      - stock_bid_ask_em    买卖盘 → 外盘/内盘/买一~五/卖一~五
      - stock_individual_fund_flow  资金流向 → 主力/超大单等净流入
    全部失败时降级为日线近似（新浪）。
    """
    from .market_catalog import parse_code
    code = code.strip()
    market, symbol = parse_code(code)
    cached = _spot_cache.get(f"quote:{code}")
    if cached is not None:
        return cached

    # 港股/美股：无实时东财数据，直接走日线近似
    if market != "a":
        rt = _fallback_from_daily_market(market, symbol)
        result = {
            "code": code,
            "market": market,
            "name": resolve_name(code),
            "time": "日线近似",
            "realtime": rt,
            "info": {},
            "fund_flow": {},
        }
        _spot_cache.set(f"quote:{code}", result)
        return result

    realtime = None
    info = None
    fund_flow = None
    source_time = "日线近似"

    # 0. 东财连通性快速检测：一次轻量请求（~0.4s），失败直接走降级，
    #    避免 spot_em 分页拉全市场在不可用时拖慢 10+ 秒
    if _fail_cache.get("em_down") is not None:
        em_ok = False
    else:
        em_ok = _check_em_reachable()
        if not em_ok:
            _fail_cache.set("em_down", True)

    # 1. 实时快照（全表缓存共享，避免每股每次拉全市场）
    if em_ok:
        if _fail_cache.get("spot_em_fail") is not None:
            spot = None
        else:
            spot = _spot_cache.get("spot_em")
            if spot is None:
                spot = _fetch_spot_em()
                if spot is not None:
                    _spot_cache.set("spot_em", spot)
                else:
                    _fail_cache.set("spot_em_fail", True)
        if spot is not None and code in spot:
            realtime = _extract_spot(spot[code], code)
            source_time = "实时"
    else:
        spot = None

    # 2. 买卖盘（外盘/内盘）
    if realtime is not None and em_ok:
        bid_ask = _fetch_bid_ask(code)
        if bid_ask:
            realtime.update(bid_ask)

    # 3. 个股信息（行业/上市时间/股本，缓存 24h；失败缓存 60s）
    if realtime is not None and em_ok:
        if _fail_cache.get(f"info_fail:{code}") is not None:
            info = None
        else:
            info = _info_cache.get(f"info:{code}")
            if info is None:
                info = _fetch_info(code)
                if info:
                    _info_cache.set(f"info:{code}", info)
                else:
                    _fail_cache.set(f"info_fail:{code}", True)

    # 4. 资金流向（缓存 10min；失败缓存 60s）
    if em_ok:
        if _fail_cache.get(f"fund_fail:{code}") is not None:
            fund_flow = None
        else:
            fund_flow = _fund_cache.get(f"fund:{code}")
            if fund_flow is None:
                fund_flow = _fetch_fund_flow(code)
                if fund_flow:
                    _fund_cache.set(f"fund:{code}", fund_flow)
                else:
                    _fail_cache.set(f"fund_fail:{code}", True)

    # 5. 全部失败 → 日线近似降级
    if realtime is None:
        realtime = _fallback_from_daily(code)

    result = {
        "code": code,
        "market": "a",
        "name": resolve_name(code),
        "time": source_time,
        "realtime": realtime,
        "info": info or {},
        "fund_flow": fund_flow or {},
    }
    _spot_cache.set(f"quote:{code}", result)
    return result


def _check_em_reachable() -> bool:
    """东财连通性快速检测：一次轻量请求判断东财是否可用。

    东财不可用（VPN 分流失败 / 网络受限）时快速返回 False，
    避免后续 spot_em 分页拉全市场拖慢 10+ 秒。
    """
    try:
        import requests
        # 用与 spot_em 相同的域名 + 分页参数检测（82.push2 与 push2 可能状态不同）
        r = requests.get(
            "https://82.push2.eastmoney.com/api/qt/clist/get",
            params={"pn": "1", "pz": "2", "po": "1", "np": "1",
                    "fltt": "2", "invt": "2", "fid": "f3",
                    "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
                    "fields": "f2,f12,f14"},
            timeout=4,
        )
        if r.status_code != 200:
            return False
        # 确认返回的是 JSON 而非错误
        return '"data"' in r.text
    except Exception:
        return False


def _fetch_spot_em() -> Optional[dict]:
    """拉全市场实时快照，返回 {code: {字段...}}"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df.empty or "代码" not in df.columns:
            return None
        result = {}
        for _, row in df.iterrows():
            code = str(row.get("代码", "")).strip()
            if len(code) != 6:
                continue
            result[code] = {
                "price": _num(row, "最新价"),
                "pct": _num(row, "涨跌幅"),
                "change": _num(row, "涨跌额"),
                "open": _num(row, "今开"),
                "high": _num(row, "最高"),
                "low": _num(row, "最低"),
                "prev_close": _num(row, "昨收"),
                "volume": _num(row, "成交量"),
                "amount": _num(row, "成交额"),
                "turnover_rate": _num(row, "换手率"),
                "volume_ratio": _num(row, "量比"),
                "pe_dynamic": _num(row, "市盈率-动态"),
                "pb": _num(row, "市净率"),
                "total_mv": _num(row, "总市值"),
                "circ_mv": _num(row, "流通市值"),
                "source": "em",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        return result
    except Exception as e:
        logger.warning(f"spot_em 拉取失败: {e}")
        return None


def _extract_spot(s: dict, code: str) -> dict:
    return dict(s)


def _fetch_bid_ask(code: str) -> Optional[dict]:
    """买卖盘：外盘/内盘 + 买卖五档"""
    try:
        import akshare as ak
        df = ak.stock_bid_ask_em(symbol=code)
        if df.empty:
            return None
        result = {}
        for _, row in df.iterrows():
            item = str(row.get("item", ""))
            value = row.get("value", None)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if "卖" in item and "量" in item:
                result["ask_vols"] = result.get("ask_vols", []) + [value]
            elif "卖" in item:
                result["ask_prices"] = result.get("ask_prices", []) + [value]
            elif "买" in item and "量" in item:
                result["bid_vols"] = result.get("bid_vols", []) + [value]
            elif "买" in item:
                result["bid_prices"] = result.get("bid_prices", []) + [value]
            elif "外盘" in item:
                result["outer_vol"] = value
            elif "内盘" in item:
                result["inner_vol"] = value
            elif "均价" in item:
                result["avg_price"] = value
            elif "涨停" in item:
                result["limit_up"] = value
            elif "跌停" in item:
                result["limit_dn"] = value
        return result or None
    except Exception as e:
        logger.warning(f"bid_ask {code} 拉取失败: {e}")
        return None


def _fetch_info(code: str) -> Optional[dict]:
    """个股基本信息：行业/上市时间/股本"""
    try:
        import akshare as ak
        df = ak.stock_individual_info_em(symbol=code)
        if df.empty:
            return None
        result = {}
        for _, row in df.iterrows():
            item = str(row.get("item", ""))
            value = row.get("value", None)
            if item == "行业":
                result["industry"] = value
            elif item == "上市时间":
                result["listing_date"] = str(value)
            elif item == "总股本":
                result["total_shares"] = _safe_float(value)
            elif item == "流通股":
                result["float_shares"] = _safe_float(value)
            elif item == "总市值":
                result["total_mv"] = _safe_float(value)
            elif item == "流通市值":
                result["circ_mv"] = _safe_float(value)
        return result or None
    except Exception as e:
        logger.warning(f"info {code} 拉取失败: {e}")
        return None


def _fetch_fund_flow(code: str) -> Optional[dict]:
    """资金流向：主力/超大单/大单/中单/小单净流入"""
    try:
        from src.data.fetcher import DataFetcher
        df = DataFetcher().fund_flow(code)
        if df is None or df.empty:
            return None
        # 取最新一行
        last = df.iloc[-1]
        date_col = None
        for c in ["日期", "date", "时间"]:
            if c in df.columns:
                date_col = c
                break

        def _flow(col_names):
            for c in col_names:
                if c in df.columns:
                    return _safe_float(last[c])
            return None

        return {
            "date": str(last[date_col]) if date_col else "",
            "main_net_inflow": _flow(["主力净流入-净额", "主力净流入"]),
            "super_large_net": _flow(["超大单净流入-净额", "超大单净流入"]),
            "large_net": _flow(["大单净流入-净额", "大单净流入"]),
            "medium_net": _flow(["中单净流入-净额", "中单净流入"]),
            "small_net": _flow(["小单净流入-净额", "小单净流入"]),
        }
    except Exception as e:
        logger.warning(f"fund_flow {code} 拉取失败: {e}")
        return None


def _fallback_from_daily(code: str) -> dict:
    """日线近似行情（东财不可用时的降级）

    新浪源补齐：现价/涨跌/换手率/量比/市值/PE/PB/每股收益。
    """
    try:
        from datetime import date, timedelta
        fetcher = _get_fetcher()
        # 只拉最近 120 天（量比需前 5 日均量）
        start = (date.today() - timedelta(days=160)).strftime("%Y-%m-%d")
        df = fetcher.daily_bars(code, start=start, end="")
        if df.empty or len(df) < 6:
            df = fetcher.daily_bars(code, start="", end="")
        if df.empty:
            return {"price": None, "source": "none", "updated_at": ""}
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        prev_close = prev["close"]
        pct = (last["close"] / prev_close - 1) * 100 if prev_close else 0
        # 换手率：新浪源是比率（0.0019=0.19%），东财是百分比（0.19=0.19%）
        # 统一转成百分比数值（如 0.19）
        turnover_raw = _safe_float(last.get("turnover_rate")) or \
            _safe_float(last.get("turnover")) or None
        turnover = None
        if turnover_raw is not None:
            turnover = round(turnover_raw * 100, 2) if turnover_raw < 1 else round(turnover_raw, 2)
        # 量比 = 当日量 / 前 5 日均量（标准算法）
        volume_ratio = _calc_volume_ratio(df)
        # 市值 = close × 流通股本（新浪日线有 outstanding_share）
        shares = _safe_float(last.get("outstanding_share"))
        circ_mv = (float(last["close"]) * shares) if shares else None
        total_mv = circ_mv  # 无总股本数据时用流通市值近似
        # 财务数据（PE/PB/EPS/ROE，新浪财务摘要）
        fin = _get_financial(code)
        price = float(last["close"])
        pe_dynamic = None
        pb = None
        if fin:
            eps = fin.get("eps")
            if eps:
                pe_dynamic = round(price / eps, 2)
            bps = fin.get("bps")
            if bps:
                pb = round(price / bps, 2)
        return {
            "price": round(price, 2),
            "pct": round(float(pct), 2),
            "change": round(float(last["close"] - prev_close), 2),
            "open": round(float(last["open"]), 2),
            "high": round(float(last["high"]), 2),
            "low": round(float(last["low"]), 2),
            "prev_close": round(float(prev_close), 2),
            "volume": _safe_float(last.get("volume")),
            "amount": _safe_float(last.get("amount")),
            "turnover_rate": turnover,
            "volume_ratio": round(volume_ratio, 2) if volume_ratio else None,
            "circ_mv": circ_mv,
            "total_mv": total_mv,
            "pe_dynamic": pe_dynamic,
            "pb": pb,
            "eps": fin.get("eps") if fin else None,
            "roe": fin.get("roe") if fin else None,
            "source": "sina_daily",
            "updated_at": str(last.get("date", "")),
        }
    except Exception as e:
        logger.warning(f"日线近似行情失败: {e}")
        return {"price": None, "source": "none", "updated_at": ""}


def _fallback_from_daily_market(market: str, symbol: str) -> dict:
    """港股/美股的日线近似行情

    现价/涨跌/开高低收/成交额（HK有，US无）/量比；
    换手率/PE/PB/市值/EPS 港股美股日线无 → None。
    """
    try:
        df = fetch_daily_df(market, symbol)
        if df.empty:
            return {"price": None, "source": "none", "updated_at": ""}
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        prev_close = prev["close"]
        pct = (last["close"] / prev_close - 1) * 100 if prev_close else 0
        return {
            "price": round(float(last["close"]), 2),
            "pct": round(float(pct), 2),
            "change": round(float(last["close"] - prev_close), 2),
            "open": round(float(last["open"]), 2),
            "high": round(float(last["high"]), 2),
            "low": round(float(last["low"]), 2),
            "prev_close": round(float(prev_close), 2),
            "volume": _safe_float(last.get("volume")),
            "amount": _safe_float(last.get("amount")),
            "turnover_rate": None,
            "volume_ratio": None,
            "source": "sina_daily",
            "updated_at": str(last.get("date", "")),
        }
    except Exception as e:
        logger.warning(f"{market} 日线近似行情失败: {e}")
        return {"price": None, "source": "none", "updated_at": ""}


def _calc_volume_ratio(df: pd.DataFrame) -> Optional[float]:
    """量比 = 当日成交量 / 前 5 日均量（不含当日）"""
    try:
        vols = pd.to_numeric(df["volume"], errors="coerce")
        if len(vols) < 6:
            return None
        today = vols.iloc[-1]
        avg5 = vols.iloc[-6:-1].mean()
        if avg5 <= 0:
            return None
        return float(today / avg5)
    except Exception:
        return None


_fin_cache = TTLCache(24 * 3600)


def _get_financial(code: str) -> Optional[dict]:
    """新浪财务摘要（缓存 24h）→ {eps, bps, roe, net_profit}"""
    cached = _fin_cache.get(f"fin:{code}")
    if cached is not None:
        return cached
    if _fail_cache.get(f"fin_fail:{code}") is not None:
        return None
    try:
        import akshare as ak
        df = ak.stock_financial_abstract(symbol=code)
        if df.empty or len(df.columns) < 3:
            return None
        indicators = df.iloc[:, 1].tolist()
        latest_col = 2  # 最新报告期（第 3 列）

        def _find(keyword):
            for i, name in enumerate(indicators):
                if keyword in str(name):
                    v = df.iloc[i, latest_col]
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
            return None

        result = {
            "eps": _find("每股收益"),
            "bps": _find("每股净资产"),
            "roe": _find("净资产收益率"),
            "net_profit": _find("归母净利润"),
        }
        if result["eps"] is None and result["net_profit"] is None:
            return None
        _fin_cache.set(f"fin:{code}", result)
        return result
    except Exception as e:
        logger.warning(f"财务数据 {code} 获取失败: {e}")
        _fail_cache.set(f"fin_fail:{code}", True)
        return None


# ============================================================================
# 分析 / 技能 / RL / LLM
# ============================================================================

def get_analysis(code: str, days: int = 500) -> dict:
    """单股深度分析（规则引擎 + 技能汇总，带 10 分钟缓存）"""
    cache_key = f"analyze:{code}:{days}"
    cached = _fund_cache.get(cache_key)
    if cached is not None:
        return cached
    from .market_catalog import parse_code
    market, symbol = parse_code(code)
    # 多市场：拉一次日线注入分析工具（A股仍走默认路径）
    df = fetch_daily_df(market, symbol) if market != "a" else None
    from src.tools.analyze_tool import analyze_stock
    try:
        result = analyze_stock(code, days=days, market=market, df=df)
        result["market"] = market
        if "error" not in result:
            _record_history(code, result.get("name", ""))
            _fund_cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.warning(f"analyze {code} 失败: {e}")
        return {"error": str(e), "code": code, "market": market}


def get_skills_analysis(code: str) -> dict:
    """本地策略分析（11 技能 + 5 维评分卡 + 统计预测，带 10 分钟缓存）"""
    cache_key = f"skills:{code}"
    cached = _fund_cache.get(cache_key)
    if cached is not None:
        return cached
    from .market_catalog import parse_code
    market, symbol = parse_code(code)
    df = fetch_daily_df(market, symbol) if market != "a" else None
    from src.tools.llm_skills_tool import skills_analyze
    try:
        result = skills_analyze(code, market=market, df=df)
        result["market"] = market
        _fund_cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.warning(f"skills {code} 失败: {e}")
        return {"error": str(e), "code": code, "market": market}


def get_llm_analysis(code: str, api_key: Optional[str] = None) -> dict:
    """LLM 深度分析（可能较慢，前端应显式触发）

    读取 LLM 配置：开关关闭时直接返回 disabled（不调 LLM）；
    api_base / model 从配置取。
    """
    import os
    from .market_catalog import parse_code
    market, symbol = parse_code(code)
    cfg = get_llm_config()
    if not cfg.get("enabled"):
        return {"action": "hold", "confidence": 0.5, "disabled": True,
                "reason": "LLM 已关闭", "code": code, "market": market}
    key = api_key or cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
    df = fetch_daily_df(market, symbol) if market != "a" else None
    from src.tools.llm_skills_tool import llm_analyze
    try:
        result = llm_analyze(code, api_key=key, market=market, df=df,
                             api_base=cfg.get("api_base"), model=cfg.get("model"))
        result["market"] = market
        return result
    except Exception as e:
        logger.warning(f"llm {code} 失败: {e}")
        return {"error": str(e), "code": code, "market": market}


def get_rl_prediction(code: str, force_refresh: bool = False) -> dict:
    """RL 模型预测（带 10 分钟缓存，避免每次重新准备数据）

    仅 A 股支持 RL（港股/美股返回未训练）。
    """
    from .market_catalog import parse_code
    market, symbol = parse_code(code)
    if market != "a":
        return {"action": "hold", "confidence": 0.5, "untrained": True,
                "reason": "RL 暂不支持港股/美股", "code": code,
                "is_trained": False, "market": market}
    cache_key = f"rl:{code}"
    if not force_refresh:
        cached = _fund_cache.get(cache_key)  # 复用 600s 缓存实例
        if cached is not None:
            return cached
    from src.tools.rl_tool import get_rl_prediction as _rl
    try:
        result = _rl(symbol, window=int(get_rl_config().get("window", 60)))
        result["code"] = code
        result["is_trained"] = not result.get("untrained", True)
        result["market"] = "a"
        _fund_cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.warning(f"rl {code} 失败: {e}")
        return {"error": str(e), "code": code, "market": "a"}


def predict_stocks(codes: list, use_llm: bool = True, api_key: Optional[str] = None) -> dict:
    """三路融合预测

    读取 LLM 配置：开关关闭时自动把 use_llm 降级为 False（本地增强策略）；
    api_base / model 从配置取。
    """
    import os
    cfg = get_llm_config()
    key = api_key or cfg.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
    if use_llm and not cfg.get("enabled"):
        use_llm = False
    from src.tools.predict_tool import predict_stocks as _predict
    return _predict(codes, api_key=key, use_llm=use_llm,
                    api_base=cfg.get("api_base"), model=cfg.get("model"))


# ============================================================================
# 历史 / 自选股
# ============================================================================

def _record_history(code: str, name: str = ""):
    try:
        _get_interested().record(code, name or resolve_name(code))
    except Exception as e:
        logger.warning(f"记录历史失败: {e}")


def _is_trained(code: str) -> bool:
    """检查该股票是否已有训练好的 RL 模型（快速，读文件系统）

    仅 A 股支持 RL（模型文件名含冒号在 Windows 非法）。
    """
    from .market_catalog import parse_code
    market, symbol = parse_code(code)
    if market != "a":
        return False
    from pathlib import Path
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / f"{symbol}_ppo.zip"
    return model_path.exists()


def _annotate(items: list) -> list:
    """补充 name 与 is_trained 字段"""
    for it in items:
        if not it.get("name"):
            it["name"] = resolve_name(it["code"])
        it["is_trained"] = _is_trained(it["code"])
    return items


def get_history(limit: int = 50) -> dict:
    """历史查询列表"""
    try:
        items = _get_interested().all()
        return {"items": _annotate(items)[:limit]}
    except Exception as e:
        logger.warning(f"history 失败: {e}")
        return {"items": [], "error": str(e)}


def get_watchlist() -> dict:
    """自选股列表（config/stocks.yaml 种子 + data/watchlist.json 增删改）"""
    items = _load_watchlist()
    return {"items": _annotate(items)}


def add_watchlist(code: str) -> dict:
    """添加自选股"""
    items = _load_watchlist()
    if code in [it["code"] for it in items]:
        return {"ok": True, "item": {"code": code, "name": resolve_name(code)},
                "exists": True}
    items.append({"code": code, "name": resolve_name(code)})
    _save_watchlist(items)
    return {"ok": True, "item": items[-1], "exists": False}


def remove_watchlist(code: str) -> dict:
    """移除自选股"""
    items = _load_watchlist()
    new_items = [it for it in items if it["code"] != code]
    _save_watchlist(new_items)
    return {"ok": True}


_WATCHLIST_FILE = None


def _watchlist_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "data" / "watchlist.json"


def _load_watchlist() -> list:
    import json
    from pathlib import Path

    path = _watchlist_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    # 种子：从 config/stocks.yaml 读取
    seed = _seed_from_yaml()
    items = [{"code": c, "name": resolve_name(c)} for c in seed]
    _save_watchlist(items)
    return items


def _seed_from_yaml() -> list:
    import yaml
    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "stocks.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return [str(c).strip() for c in (cfg.get("watchlist") or []) if str(c).strip()]
    except Exception:
        return []


def _save_watchlist(items: list):
    import json
    path = _watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# ============================================================================
# 大盘状态
# ============================================================================

# ============================================================================
# 规则参数编辑（运行时覆盖，不改 config.yaml 注释）
# ============================================================================

_RULES_OVERRIDE_FILE = None


def _rules_override_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "data" / "rules_override.json"


# 默认规则参数（与 config.yaml knowledge.rules 一致）
DEFAULT_RULES_CONFIG = {
    "trend_ma_short": 5,
    "trend_ma_medium": 20,
    "trend_ma_long": 60,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "volume_surge_ratio": 2.0,
    "fusion_llm_skills": 0.40,
    "fusion_rl": 0.30,
    "fusion_rule": 0.30,
    "kelly_odds": 4.0,
    "max_position": 0.20,
}

def _skill_names() -> list:
    """动态获取已注册技能名清单"""
    from src.skills import init_skills, SkillRegistry
    try:
        init_skills()
        return [s.name for s in SkillRegistry.list_all()]
    except Exception:
        return []


def _default_skills_config() -> dict:
    """默认技能配置（fusion 默认值取自 LocalFusionEngine 常量，避免漂移）"""
    from src.agent.local_fusion import LOCAL_FUSION_DEFAULTS
    return {
        "enabled": True,
        "confidence_threshold": 0.50,
        "fusion": dict(LOCAL_FUSION_DEFAULTS),
        "skill_switches": {name: True for name in _skill_names()},
    }

# RL 训练/环境参数（扁平视图，前端可直接编辑；持久化到 data/rl_config.json）
DEFAULT_RL_CONFIG = {
    # 训练步数
    "timesteps_full": 200_000,
    "timesteps_incremental": 10_000,
    # 生命周期
    "update_interval_days": 15,
    "delete_stale_days": 60,
    "min_data_rows": 120,
    # 环境参数
    "window": 60,
    "min_hold_days": 3,
    "max_hold_days": 30,
    "commission": 0.00025,
    "stamp_tax": 0.001,
    "limit_pct": 0.10,
    # PPO 超参
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "batch_size": 64,
    "n_steps": 2048,
}


def get_rules_config() -> dict:
    """读取当前生效的规则参数（默认 + override 合并）"""
    import json
    cfg = dict(DEFAULT_RULES_CONFIG)
    path = _rules_override_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                override = json.load(f)
            cfg.update(override.get("rules", {}))
        except (json.JSONDecodeError, IOError):
            pass
    return cfg


def save_rules_config(params: dict) -> dict:
    """保存规则参数到 override 文件"""
    import json
    path = _rules_override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    override = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                override = json.load(f)
        except (json.JSONDecodeError, IOError):
            override = {}
    # 只保存合法键
    allowed = set(DEFAULT_RULES_CONFIG.keys())
    cleaned = {k: float(v) for k, v in params.items() if k in allowed and _safe_float(v) is not None}
    override["rules"] = cleaned
    with open(path, "w", encoding="utf-8") as f:
        json.dump(override, f, ensure_ascii=False, indent=2)
    return {"ok": True, "rules": cleaned}


def reset_rules_config() -> dict:
    """重置规则参数为默认"""
    from pathlib import Path
    path = _rules_override_path()
    if path.exists():
        path.unlink()
    return {"ok": True, "rules": dict(DEFAULT_RULES_CONFIG)}


def _skills_override_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "data" / "skills_config.json"


def get_skills_config() -> dict:
    """读取当前生效的技能配置（默认 + override 合并）"""
    import json
    cfg = _default_skills_config()
    path = _skills_override_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                override = json.load(f)
            if "enabled" in override:
                cfg["enabled"] = bool(override["enabled"])
            if "confidence_threshold" in override and _safe_float(override["confidence_threshold"]) is not None:
                cfg["confidence_threshold"] = float(override["confidence_threshold"])
            if isinstance(override.get("fusion"), dict):
                for k, v in override["fusion"].items():
                    if k in cfg["fusion"] and _safe_float(v) is not None:
                        cfg["fusion"][k] = float(v)
            if isinstance(override.get("skill_switches"), dict):
                for k, v in override["skill_switches"].items():
                    if k in cfg["skill_switches"]:
                        cfg["skill_switches"][k] = bool(v)
        except (json.JSONDecodeError, IOError):
            pass
    return cfg


def save_skills_config(params: dict) -> dict:
    """保存技能配置到 data/skills_config.json（白名单校验，真正生效）"""
    import json
    path = _skills_override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    default = _default_skills_config()
    override = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                override = json.load(f)
        except (json.JSONDecodeError, IOError):
            override = {}
    if "enabled" in params:
        override["enabled"] = bool(params["enabled"])
    if "confidence_threshold" in params and _safe_float(params["confidence_threshold"]) is not None:
        override["confidence_threshold"] = float(params["confidence_threshold"])
    fusion = params.get("fusion")
    if isinstance(fusion, dict):
        base_fusion = dict(default["fusion"])
        for k, v in fusion.items():
            if k in base_fusion and _safe_float(v) is not None:
                base_fusion[k] = float(v)
        override["fusion"] = base_fusion
    switches = params.get("skill_switches")
    if isinstance(switches, dict):
        base_sw = dict(default["skill_switches"])
        for k, v in switches.items():
            if k in base_sw:
                base_sw[k] = bool(v)
        override["skill_switches"] = base_sw
    with open(path, "w", encoding="utf-8") as f:
        json.dump(override, f, ensure_ascii=False, indent=2)
    return {"ok": True, "skills": get_skills_config()}


def reset_skills_config() -> dict:
    """重置技能配置为默认"""
    path = _skills_override_path()
    if path.exists():
        path.unlink()
    return {"ok": True, "skills": _default_skills_config()}


# ============================================================================
# RL 训练/环境参数编辑（持久化 data/rl_config.json）
# ============================================================================

def _rl_override_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "data" / "rl_config.json"


def get_rl_config() -> dict:
    """读取当前生效的 RL 参数（默认 + override 合并，扁平键）"""
    import json
    cfg = dict(DEFAULT_RL_CONFIG)
    path = _rl_override_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                override = json.load(f)
            cfg.update({k: v for k, v in override.items() if k in DEFAULT_RL_CONFIG})
        except (json.JSONDecodeError, IOError):
            pass
    return cfg


def save_rl_config(params: dict) -> dict:
    """保存 RL 参数到 data/rl_config.json（白名单校验）"""
    import json
    path = _rl_override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    override = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                override = json.load(f)
        except (json.JSONDecodeError, IOError):
            override = {}
    allowed = set(DEFAULT_RL_CONFIG.keys())
    cleaned = {}
    for k, v in params.items():
        if k in allowed and _safe_float(v) is not None:
            cleaned[k] = float(v)
    override.update(cleaned)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(override, f, ensure_ascii=False, indent=2)
    return {"ok": True, "rl": cleaned}


def reset_rl_config() -> dict:
    """重置 RL 参数为默认"""
    path = _rl_override_path()
    if path.exists():
        path.unlink()
    return {"ok": True, "rl": dict(DEFAULT_RL_CONFIG)}


def rl_train_config() -> dict:
    """扁平 RL 配置 → train_single_stock 使用的嵌套 config

    Returns:
        {timesteps_full, timesteps_incremental, update_interval_days,
         delete_stale_days, min_data_rows, ppo_kwargs{...}, window,
         min_hold_days, max_hold_days, commission, stamp_tax, limit_pct}
    """
    cfg = get_rl_config()
    return {
        "timesteps_full": cfg["timesteps_full"],
        "timesteps_incremental": cfg["timesteps_incremental"],
        "update_interval_days": cfg["update_interval_days"],
        "delete_stale_days": cfg["delete_stale_days"],
        "min_data_rows": cfg["min_data_rows"],
        "window": cfg["window"],
        "min_hold_days": cfg["min_hold_days"],
        "max_hold_days": cfg["max_hold_days"],
        "commission": cfg["commission"],
        "stamp_tax": cfg["stamp_tax"],
        "limit_pct": cfg["limit_pct"],
        "ppo_kwargs": {
            "learning_rate": cfg["learning_rate"],
            "gamma": cfg["gamma"],
            "gae_lambda": cfg["gae_lambda"],
            "clip_range": cfg["clip_range"],
            "ent_coef": cfg["ent_coef"],
            "batch_size": cfg["batch_size"],
            "n_steps": cfg["n_steps"],
        },
    }


# ============================================================================
# 市场热门 / 列表（多市场）
# ============================================================================

_hot_cache = TTLCache(120)


def get_market_hot(market: str, top: int = 20) -> dict:
    """某市场的热门股票列表

    数据源分层：
      A股: 新浪 spot → 扫描器 → 内置池
      港股/美股: 东财热门 → 内置池 → 名称列表
    """
    from .market_catalog import to_canonical, market_name
    market = market if market in ("a", "hk", "us") else "a"
    cache_key = f"hot:{market}:{top}"
    cached = _hot_cache.get(cache_key)
    if cached is not None:
        return cached

    items, source = [], "realtime"
    if market == "a":
        items, source = _a_hot(top)
    elif market == "hk":
        items, source = _hk_hot(top)
    elif market == "us":
        items, source = _us_hot(top)

    # 补充 market_tag 字段
    tag = market_name(market)
    for it in items:
        it.setdefault("market_tag", tag)
        if "code" in it:
            it["code"] = to_canonical(market, str(it.get("_symbol") or it["code"]))

    result = {"items": items, "source": source, "market": market}
    _hot_cache.set(cache_key, result)
    return result


def _a_hot(top: int) -> tuple:
    """A股热门：内置池并行拉取

    避免调用 stock_zh_a_spot（demjson/MiniRacer 在 Windows 上会崩溃，
    且首次拉全市场 5000+ 行 15-20s）。改用内置池并行，~3s 返回。
    """
    from .market_catalog import A_HOT_FALLBACK
    items = _parallel_pool_fallback("a", A_HOT_FALLBACK, top)
    return items, "fallback"


def _hk_hot(top: int) -> tuple:
    """港股热门：东财热度榜 → 内置池"""
    try:
        import akshare as ak
        df = ak.stock_hk_hot_rank_em()
        if not df.empty and len(df.columns) > 3:
            cols = df.columns.tolist()
            # 列顺序常见: 当前排名/代码/名称/最新价/涨跌幅
            code_col = next((i for i, c in enumerate(cols) if "代码" in str(c)), 1)
            name_col = next((i for i, c in enumerate(cols) if "名称" in str(c)), 2)
            pct_col = next((i for i, c in enumerate(cols) if "涨跌幅" in str(c)), 4)
            price_col = next((i for i, c in enumerate(cols) if "最新价" in str(c)), 3)
            items = []
            for _, row in df.head(top).iterrows():
                code = str(row[code_col]).replace("0" * 5, "").zfill(5)
                if code.startswith("0") and len(code) == 5:
                    code = code.zfill(5)
                items.append({
                    "code": code, "_symbol": code,
                    "name": str(row[name_col]),
                    "price": _safe_float(row[price_col]),
                    "pct": _safe_float(row[pct_col]),
                })
            if items:
                return items, "realtime"
    except Exception as e:
        logger.warning(f"港股热门失败: {e}")

    return _pool_hot("hk", top), "fallback"


def _us_hot(top: int) -> tuple:
    """美股热门：东财热门 → 内置池"""
    try:
        import akshare as ak
        df = ak.stock_us_famous_spot_em()
        if not df.empty and len(df.columns) > 3:
            cols = df.columns.tolist()
            code_col = next((i for i, c in enumerate(cols) if "代码" in str(c) or "symbol" in str(c).lower()), 0)
            name_col = next((i for i, c in enumerate(cols) if "名称" in str(c) or "name" in str(c).lower()), 1)
            pct_col = next((i for i, c in enumerate(cols) if "涨跌幅" in str(c)), 4)
            price_col = next((i for i, c in enumerate(cols) if "最新价" in str(c)), 3)
            items = []
            for _, row in df.head(top).iterrows():
                items.append({
                    "code": str(row[code_col]), "_symbol": str(row[code_col]),
                    "name": str(row[name_col]),
                    "price": _safe_float(row[price_col]),
                    "pct": _safe_float(row[pct_col]),
                })
            if items:
                return items, "realtime"
    except Exception as e:
        logger.warning(f"美股热门失败: {e}")

    return _pool_hot("us", top), "fallback"


def _pool_hot(market: str, top: int) -> list:
    """内置池兜底：并行拉日线算现价/涨跌"""
    from .market_catalog import get_pool
    return _parallel_pool_fallback(market, get_pool(market), top)


def _parallel_pool_fallback(market: str, pool: list, top: int) -> list:
    """并行拉取池内股票的日线近似行情（避免串行 25s+）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    items = []
    try:
        # 并发 3：akshare 有全局锁，过高的并发无法提升吞吐且增加 MiniRacer 风险
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {}
            for code, name in pool[:top]:
                fut = ex.submit(_fallback_from_daily_market, market, code)
                futures[fut] = (code, name)
            for fut in as_completed(futures):
                code, name = futures[fut]
                try:
                    rt = fut.result(timeout=15)
                except Exception:
                    rt = {"price": None, "pct": None}
                items.append({
                    "code": code, "_symbol": code, "name": name,
                    "price": rt.get("price"), "pct": rt.get("pct"),
                })
        # 保持池内顺序
        order = {c: i for i, (c, _) in enumerate(pool[:top])}
        items.sort(key=lambda x: order.get(x["code"], 99))
    except Exception as e:
        logger.warning(f"并行拉取池行情失败: {e}")
    return items


def get_market_list(market: str, q: str = "", limit: int = 20) -> dict:
    """市场浏览/搜索：A股用 resolver，港股/美股用内置池"""
    from .market_catalog import get_pool, to_canonical
    market = market if market in ("a", "hk", "us") else "a"
    if market == "a":
        return search_stocks(q, limit) if q else {"items": [], "market": market}
    pool = get_pool(market)
    if q:
        pool = [(c, n) for c, n in pool if q.lower() in n.lower() or q.lower() in c.lower()]
    items = [{"code": to_canonical(market, c), "name": n} for c, n in pool[:limit]]
    return {"items": items, "market": market}


def get_market_state() -> dict:
    """大盘状态 + 建议仓位"""
    try:
        from src.scanner.market_watch import MarketWatch
        from src.data.fetcher import DataFetcher
        market = MarketWatch(fetcher=DataFetcher())
        state = market.update()
        return {
            "market_state": state.get("state", "unknown"),
            "ret_20d": state.get("ret_20d", 0),
            "suggested_position": market.suggest_position_level(),
        }
    except Exception as e:
        logger.warning(f"market state 失败: {e}")
        return {"market_state": "unknown", "ret_20d": 0, "suggested_position": 0,
                "error": str(e)}


# ============================================================================
# LLM 配置（api_base / model / api_key / enabled），持久化到 data/llm_config.json
# 委托 src/llm_config.py —— 全局唯一事实来源，整个软件共用同一开关。
# ============================================================================

def get_llm_config() -> dict:
    """读取全局 LLM 配置（默认值 + 用户覆盖合并）

    enabled 未显式保存过时默认 = 是否有 API Key。
    """
    from src.llm_config import llm_config
    return llm_config()


def save_llm_config(params: dict) -> dict:
    """保存 LLM 配置到 data/llm_config.json，并重置聊天 Agent 使新配置生效"""
    from src.llm_config import save_config
    save_config(params)
    # 配置变更 → 重置聊天 Agent（下次聊天用新 key/model/开关）
    try:
        from src.webui import chat
        chat.reset_agent()
    except Exception:
        pass
    return get_llm_config()


def test_llm_connection(params: dict) -> dict:
    """测试 LLM 连接：发一个极短的 chat 请求"""
    import os
    api_base = (params.get("api_base") or "").strip() or "https://api.deepseek.com"
    model = (params.get("model") or "").strip() or "deepseek-v4-flash"
    api_key = (params.get("api_key") or "").strip() or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return {"ok": False, "error": "未提供 API Key（请填写，或设置环境变量 DEEPSEEK_API_KEY）"}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base, timeout=15)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        reply = (resp.choices[0].message.content or "").strip()[:80]
        return {"ok": True, "model": model, "reply": reply}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def get_llm_status(api_key: Optional[str] = None) -> dict:
    cfg = get_llm_config()
    if api_key:
        cfg["has_api_key"] = True
    return {
        "has_api_key": bool(cfg["has_api_key"]),
        "enabled": bool(cfg["enabled"]),
        "model": cfg["model"],
        "api_base": cfg["api_base"],
    }


# ============================================================================
# 工具
# ============================================================================

def _num(row, col):
    """从行中安全取数值"""
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _safe_float(v):
    try:
        f = float(v)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
