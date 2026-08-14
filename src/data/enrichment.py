"""data/enrichment.py — 外部数据增强

将大盘指数、板块、资金流向、换手率等外部数据合并到个股 DataFrame，
为 RL 模型提供更丰富的特征维度。
"""

from typing import Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def enrich_all(df: pd.DataFrame, code: str,
               index_df: Optional[pd.DataFrame] = None,
               fund_flow_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """一次性添加所有外部增强特征

    Args:
        df: 个股日线 DataFrame（必须含 date, close, volume, amount）
        code: 股票代码
        index_df: 对应大盘指数日线（可选，不传则用 0 填充）
        fund_flow_df: 个股资金流向数据（可选，不传则用 0 填充）

    Returns:
        添加了外部特征列的 DataFrame
    """
    df = df.copy()

    # ---- 1. 大盘相关 ----
    df = add_index_features(df, index_df)

    # ---- 2. 资金流向 ----
    df = add_fund_flow_features(df, fund_flow_df)

    # ---- 3. 换手率 ----
    df = add_turnover_features(df)

    return df


# ============================================================================
# 大盘指数特征
# ============================================================================

def add_index_features(df: pd.DataFrame,
                       index_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """添加大盘相关特征

    新增列:
        index_return_5d  — 大盘近5日涨跌幅%
        stock_vs_index_20d — 个股相对大盘20日超额收益%
        index_trend_20d  — 大盘20日均线方向 (close/MA20 - 1)
    """
    if index_df is None or index_df.empty:
        df["index_return_5d"] = 0.0
        df["stock_vs_index_20d"] = 0.0
        df["index_trend_20d"] = 0.0
        return df

    # 对齐日期
    index_df = index_df.copy()
    if "date" in index_df.columns:
        index_df["date"] = pd.to_datetime(index_df["date"])
    if "close" not in index_df.columns:
        df["index_return_5d"] = 0.0
        df["stock_vs_index_20d"] = 0.0
        df["index_trend_20d"] = 0.0
        return df

    # 大盘日涨跌幅
    index_df["idx_change"] = index_df["close"].pct_change(1) * 100
    # 大盘5日涨跌幅
    index_df["idx_return_5d"] = index_df["close"].pct_change(5) * 100
    # 大盘20日涨跌幅
    index_df["idx_return_20d"] = index_df["close"].pct_change(20) * 100
    # 大盘趋势（价格 vs 20日均线）
    index_df["idx_ma20"] = index_df["close"].rolling(20).mean()
    index_df["idx_trend_20d"] = (index_df["close"] / (index_df["idx_ma20"] + 1e-10) - 1) * 100

    # 合并到个股 df
    idx_map = index_df.set_index("date")[["idx_return_5d", "idx_return_20d", "idx_trend_20d"]]
    df = df.join(idx_map, on="date", how="left")

    # 前向填充（指数数据可能的缺失日）
    df["idx_return_5d"] = df["idx_return_5d"].ffill().fillna(0)
    df["idx_return_20d"] = df["idx_return_20d"].ffill().fillna(0)
    df["idx_trend_20d"] = df["idx_trend_20d"].ffill().fillna(0)

    # 个股涨跌幅（如果还没算的话）
    if "price_change" not in df.columns:
        df["price_change"] = df["close"].pct_change(1) * 100

    # 个股20日涨跌幅
    stock_ret_20d = df["close"].pct_change(20) * 100

    # 重命名大盘列
    df["index_return_5d"] = df["idx_return_5d"]
    df["index_trend_20d"] = df["idx_trend_20d"]
    df["stock_vs_index_20d"] = stock_ret_20d - df["idx_return_20d"]

    # 清理临时列
    for col in ["idx_return_5d", "idx_return_20d", "idx_trend_20d"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


# ============================================================================
# 资金流向特征
# ============================================================================

def add_fund_flow_features(df: pd.DataFrame,
                           fund_flow_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """添加资金流向特征

    akshare stock_individual_fund_flow 返回列:
        日期, 主力净流入-净额, 超大单净流入-净额, 大单净流入-净额,
        中单净流入-净额, 小单净流入-净额

    新增列:
        main_flow_pct  — 主力净流入 / 成交额 (%)
        main_flow_5d   — 近5日主力净流入累计 / 近5日成交额累计 (%)
    """
    if fund_flow_df is None or fund_flow_df.empty:
        df["main_flow_pct"] = 0.0
        df["main_flow_5d"] = 0.0
        return df

    ff = fund_flow_df.copy()

    # 标准化列名（akshare 中文列名）
    date_col = _find_column(ff, ["日期", "date", "时间"])
    flow_col = _find_column(ff, ["主力净流入-净额", "主力净流入", "main_flow"])
    amount_col = _find_column(ff, ["成交额", "amount", "成交金额"])

    if date_col is None:
        logger.warning("资金流向数据缺少日期列，跳过")
        df["main_flow_pct"] = 0.0
        df["main_flow_5d"] = 0.0
        return df

    ff[date_col] = pd.to_datetime(ff[date_col])
    ff = ff.sort_values(date_col)

    # 计算主力净流入占比
    if flow_col is not None:
        ff["_flow"] = pd.to_numeric(ff[flow_col], errors="coerce").fillna(0)
    else:
        ff["_flow"] = 0.0

    # 成交额用于归一化
    if amount_col is not None and "amount" not in df.columns:
        df["_amount_from_flow"] = 0.0
    if amount_col is not None:
        ff["_amt"] = pd.to_numeric(ff[amount_col], errors="coerce").fillna(1e8)
    elif "amount" in df.columns:
        # 从原始 df 中匹配成交额
        amt_map = df.set_index("date")["amount"] if "date" in df.columns else None
        if amt_map is not None:
            ff["_amt"] = ff[date_col].map(amt_map).fillna(1e8)
        else:
            ff["_amt"] = 1e8
    else:
        ff["_amt"] = 1e8

    ff["main_flow_pct"] = ff["_flow"] / (ff["_amt"] + 1e-8) * 100

    # 5日主力净流入累计占比
    ff["_flow_5d"] = ff["_flow"].rolling(5).sum()
    ff["_amt_5d"] = ff["_amt"].rolling(5).sum()
    ff["main_flow_5d"] = ff["_flow_5d"] / (ff["_amt_5d"] + 1e-8) * 100

    # 合并到 df
    merge_cols = [date_col, "main_flow_pct", "main_flow_5d"]
    ff_sub = ff[merge_cols].rename(columns={date_col: "date"})
    ff_sub["date"] = pd.to_datetime(ff_sub["date"])

    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(ff_sub, on="date", how="left")
    df["main_flow_pct"] = df["main_flow_pct"].fillna(0)
    df["main_flow_5d"] = df["main_flow_5d"].fillna(0)

    # 清理
    for col in ["_amount_from_flow"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


# ============================================================================
# 换手率特征
# ============================================================================

def add_turnover_features(df: pd.DataFrame) -> pd.DataFrame:
    """提取/计算换手率特征

    新增列:
        turnover_rate    — 换手率%（如果数据源已有则直接使用）
        turnover_ma5     — 5日平均换手率
        turnover_change  — 换手率变化率（当日/5日均值 - 1）
    """
    # 如果数据中已有换手率
    if "turnover_rate" in df.columns:
        df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce").fillna(0)
    else:
        # 尝试从成交量和流通股本估算（粗略）
        # 大多数 A 股日线数据不含流通股本，用 0 占位
        df["turnover_rate"] = 0.0

    # 衍生特征
    df["turnover_ma5"] = df["turnover_rate"].rolling(5).mean().fillna(0)
    df["turnover_change"] = df["turnover_rate"] / (df["turnover_ma5"] + 1e-8) - 1

    return df


# ============================================================================
# 工具
# ============================================================================

def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """在 DataFrame 中查找第一个存在的候选列名"""
    for col in candidates:
        if col in df.columns:
            return col
    return None
