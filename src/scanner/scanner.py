"""scanner/scanner.py — 热门股票扫描引擎

全市场扫描主力逻辑，组合多条件筛选热门股票并打分排序。
"""

from typing import List, Dict, Optional, Set
import pandas as pd
import numpy as np
import logging

from .criteria import evaluate_all, volume_surge, price_momentum, ma_bullish, breakout_high
from src.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


# ST / 退市股票代码前缀
ST_PREFIXES = ("ST", "*ST", "SST", "S*ST", "N", "C")
EXCLUDE_CODES: Set[str] = set()


def scan_hot_stocks(top_n: int = 20, fetcher: Optional[DataFetcher] = None,
                    lookback_days: int = 120) -> List[Dict]:
    """扫描全市场，筛选热门股票

    流程：
    1. 获取全市场股票列表
    2. 过滤 ST、退市等排除项
    3. 获取每只股票的近期数据
    4. 运行筛选条件组合打分
    5. 按综合得分排序，返回 top_n

    Args:
        top_n: 返回的热门股数量
        fetcher: 数据获取器（默认新建）
        lookback_days: 获取多少天的历史数据

    Returns:
        [{"code": "...", "name": "...", "score": float, "signals": {...}}, ...]
    """
    f = fetcher or DataFetcher()

    # 1. 获取全市场股票列表
    try:
        all_stocks = f.source.all_stocks()
    except Exception as e:
        logger.warning(f"获取全市场列表失败: {e}")
        # 备选：使用预设的热门股票池
        fallback_codes = ["000001", "000002", "000858", "002415", "300750",
                          "600519", "601318", "600036", "000333", "002714"]
        return _scan_fallback(f, fallback_codes, lookback_days)

    if all_stocks.empty:
        return _scan_fallback(f, ["000001", "000002", "000858", "600519"], lookback_days)

    # 2. 过滤 + 按活跃度预排序
    candidates = []
    for _, row in all_stocks.iterrows():
        raw_code = str(row.get("代码", row.get("code", "")))
        name = str(row.get("名称", row.get("name", "")))

        if not raw_code:
            continue

        # 去掉 sh/sz 前缀保留纯数字代码
        code = raw_code.replace("sh", "").replace("sz", "")

        if code in EXCLUDE_CODES:
            continue
        if any(code.startswith(prefix) for prefix in ("bj", "BJ", "8")):
            continue  # 北交所暂不支持
        if any(name.startswith(p) for p in ST_PREFIXES):
            continue

        # 用涨跌幅绝对值作为活跃度参考
        change_pct = row.get("涨跌幅", 0)
        if change_pct is None:
            change_pct = 0
        try:
            activity = abs(float(change_pct))
        except (ValueError, TypeError):
            activity = 0

        candidates.append({"code": code, "name": name, "activity": activity})

    if not candidates:
        return _scan_fallback(f, ["000001", "000002", "000858", "600519"], lookback_days)

    # 按活跃度排序，取最活跃的前 N 只
    candidates.sort(key=lambda x: x["activity"], reverse=True)
    max_to_check = min(len(candidates), 80)
    top_candidates = candidates[:max_to_check]

    codes = [c["code"] for c in top_candidates]
    names = {c["code"]: c["name"] for c in top_candidates}

    # 3. 获取每只股票数据并评分
    from datetime import datetime, timedelta
    lookback_start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    results = []
    total = len(codes)
    for i, code in enumerate(codes[:max_to_check]):
        # 简单进度
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  ⏳ 扫描进度: {i+1}/{total}", flush=True)
        try:
            df = f.daily_bars(code, start=lookback_start, end="")
            if df.empty or len(df) < 20:
                continue
            score, signals = calculate_hot_score(df)
            results.append({
                    "code": code,
                    "name": names.get(code, ""),
                    "score": round(score, 4),
                    "signals": signals,
                    "last_close": float(df["close"].iloc[-1]),
                    "volume_ratio": float(signals.get("volume_surge", 0) or 0),
                    "momentum": float(signals.get("price_momentum", 0) or 0),
                })
        except Exception:
            continue

    # 5. 排序取 top
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def _scan_fallback(fetcher: DataFetcher, codes: List[str],
                   lookback_days: int = 120) -> List[Dict]:
    """备选扫描：使用预设股票列表"""
    from datetime import datetime, timedelta
    lookback_start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    results = []
    for code in codes:
        try:
            df = fetcher.daily_bars(code, start=lookback_start, end="")
            if df.empty or len(df) < 20:
                continue
            score, signals = calculate_hot_score(df)
            if score > 0:
                results.append({
                    "code": code,
                    "name": "",
                    "score": round(score, 4),
                    "signals": signals,
                    "last_close": float(df["close"].iloc[-1]),
                })
        except Exception:
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def calculate_hot_score(df: pd.DataFrame) -> float:
    """根据筛选条件结果计算综合热度得分

    Returns:
        (score, signals_dict)
    """
    if df.empty:
        return 0.0, {}

    criteria_results = evaluate_all(df)
    score = 0.0

    # 1. 量比突增 - 最高25分
    vol_surge_val = criteria_results.get("volume_surge", False)
    if vol_surge_val and isinstance(vol_surge_val, (int, float)):
        score += min(vol_surge_val * 10, 25)
    else:
        # 即使未达阈值也加基础分（只要有量）
        last_vol = df["volume"].iloc[-1]
        vol_ma20 = df["volume"].tail(20).mean()
        if vol_ma20 > 0:
            vol_ratio = last_vol / vol_ma20
            score += min(vol_ratio * 3, 10)  # 量比基础分

    # 2. 价格动量 - 最高20分
    mom_val = criteria_results.get("price_momentum", False)
    if mom_val and isinstance(mom_val, (int, float)):
        score += min(abs(mom_val) * 100, 20)

    # 短期趋势方向（5日涨跌幅基础分）
    if len(df) > 5:
        ret_5d = df["close"].pct_change(5).iloc[-1]
        if ret_5d > 0:
            score += min(ret_5d * 100, 10)  # 正向动量基础分

    # 3. 均线多头排列 - 最高20分
    ma_val = criteria_results.get("ma_bullish", False)
    if ma_val and isinstance(ma_val, (int, float)):
        score += ma_val * 20
    # MA5 > MA20 基础分（即使不是完整多头排列）
    if "MA5" in df.columns and "MA20" in df.columns:
        if df["MA5"].iloc[-1] > df["MA20"].iloc[-1]:
            score += 5  # 短期均线向上

    # 4. 突破新高 - 最高15分
    breakout_val = criteria_results.get("breakout_high", False)
    if breakout_val and isinstance(breakout_val, (int, float)):
        score += min(breakout_val * 3, 15)

    # 5. MACD信号 - 最高10分
    macd_sig = criteria_results.get("macd_signal", "none")
    if macd_sig == "golden_cross":
        score += 10
    elif macd_sig == "dead_cross":
        score -= 10
    # MACD柱线方向基础分
    if "MACD_HIST" in df.columns and len(df) > 2:
        if df["MACD_HIST"].iloc[-1] > df["MACD_HIST"].iloc[-2]:
            score += 3  # 红柱放大

    # 6. RSI 加分
    rsi_sig = criteria_results.get("rsi_signal", "normal")
    if rsi_sig == "oversold":
        score += 5
    elif rsi_sig == "overbought":
        score -= 5

    return round(max(score, 0), 2), criteria_results


def filter_excluded(stocks: List[str]) -> List[str]:
    """过滤 ST、退市等排除股票"""
    result = []
    for code in stocks:
        if code in EXCLUDE_CODES:
            continue
        result.append(code)
    return result
