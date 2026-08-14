"""data/pattern_cache.py — 历史技术模式缓存与相似度搜索

为 LLM 关闭时的本地统计预测提供数据基础。

核心流程：
1. 首次分析某股 → 拉 3 年日线 → 算 15 维特征向量 → 存 parquet
2. 后续分析 → 增量更新 → 余弦相似度搜索 Top-10 历史相似天
3. 从相似天的未来走势中推导统计预测

存储: data/patterns/{code}_features.parquet
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认缓存目录
DEFAULT_CACHE_DIR = "data/patterns"

# 特征维度定义（15 维）
# 每个维度归一化到相似量级，用于余弦相似度计算
FEATURE_DIMS = [
    # (列名/计算表达式, 权重, 类别)
    ("close_vs_ma5",   0.12),   # 短线偏离
    ("close_vs_ma20",  0.12),   # 中线偏离
    ("close_vs_ma60",  0.08),   # 长线偏离
    ("return_5d",      0.10),   # 5日动能
    ("return_20d",     0.08),   # 20日动能
    ("rsi_norm",       0.07),   # RSI → [0,1]
    ("kdj_k_norm",     0.05),   # KDJ_K → [0,1]
    ("boll_pct_b",     0.06),   # 布林带%B
    ("atr_pct",        0.06),   # 波动率
    ("vol_ratio_norm", 0.06),   # 量比归一化
    ("macd_hist_norm", 0.04),   # MACD柱
    ("main_flow_pct",  0.06),   # 主力资金当日
    ("main_flow_5d",   0.05),   # 主力资金5日
    ("turnover_norm",  0.03),   # 换手率
    ("trend_strength", 0.02),   # 趋势强度（均线发散度）
]

# 搜索排除最近 N 天（避免偷看未来 / 排除高度自相关的相邻日）
EXCLUDE_RECENT = 30

# 首次拉取历史数据年数
INITIAL_YEARS = 3

# 最小相似度阈值（低于此值的匹配不可靠）
MIN_SIMILARITY = 0.60

# 最少样本数（样本太少则预测不可靠）
MIN_SAMPLES = 5


# ============================================================================
# 特征计算
# ============================================================================

def compute_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV + 指标 DataFrame 中提取 15 维特征矩阵

    Args:
        df: 含全部技术指标的 DataFrame（已运行 add_all_indicators + enrich_all）

    Returns:
        特征 DataFrame，每行一个交易日，列为特征名
    """
    features = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)

    # 均线偏离
    for ma_col, feat_name in [("MA5", "close_vs_ma5"),
                               ("MA20", "close_vs_ma20"),
                               ("MA60", "close_vs_ma60")]:
        if ma_col in df.columns:
            features[feat_name] = (close / (df[ma_col].astype(float) + 1e-10) - 1) * 100
        else:
            features[feat_name] = 0.0

    # 动量
    features["return_5d"] = close.pct_change(5) * 100
    features["return_20d"] = close.pct_change(20) * 100

    # RSI 归一化
    features["rsi_norm"] = df["RSI"].astype(float).fillna(50) / 100 if "RSI" in df.columns else 0.5

    # KDJ
    features["kdj_k_norm"] = df["KDJ_K"].astype(float).fillna(50) / 100 if "KDJ_K" in df.columns else 0.5

    # 布林带%B
    features["boll_pct_b"] = df["BOLL_POSITION"].astype(float).fillna(0.5) if "BOLL_POSITION" in df.columns else 0.5

    # ATR%
    features["atr_pct"] = df["ATR_PCT"].astype(float).fillna(2.0) if "ATR_PCT" in df.columns else 2.0

    # 量比归一化 (量比在 0~3 之间浮动，/3 压到 [0,1] 附近)
    features["vol_ratio_norm"] = df["VOL_RATIO"].astype(float).fillna(1.0) / 3.0 if "VOL_RATIO" in df.columns else 0.33

    # MACD 柱归一化（除以股价）
    if "MACD_HIST" in df.columns:
        macd_hist = df["MACD_HIST"].astype(float)
        features["macd_hist_norm"] = macd_hist / (close + 1e-8) * 100
    else:
        features["macd_hist_norm"] = 0.0

    # 资金流向
    features["main_flow_pct"] = df["main_flow_pct"].astype(float).fillna(0) if "main_flow_pct" in df.columns else 0.0
    features["main_flow_5d"] = df["main_flow_5d"].astype(float).fillna(0) if "main_flow_5d" in df.columns else 0.0

    # 换手率归一化
    if "turnover_rate" in df.columns:
        turnover = df["turnover_rate"].astype(float).fillna(2.0)
        features["turnover_norm"] = turnover / 10.0  # 换手率一般在0~10%
    else:
        features["turnover_norm"] = 0.2

    # 趋势强度（均线发散度）
    if all(c in df.columns for c in ["MA5", "MA20", "MA60"]):
        mas = np.column_stack([
            df["MA5"].astype(float),
            df["MA20"].astype(float),
            df["MA60"].astype(float),
        ])
        mas_max = np.max(mas, axis=1)
        mas_min = np.min(mas, axis=1)
        features["trend_strength"] = np.where(mas_min > 0, mas_max / (mas_min + 1e-10) - 1, 0) * 100
    else:
        features["trend_strength"] = 0.0

    # 保留 close 列（供后续计算未来收益率）
    features["close"] = close.values

    # 填充前几行的 NaN（pct_change 默认产生）
    features = features.fillna(0)

    # 日期列
    if "date" in df.columns:
        features["date"] = df["date"]

    return features


def standardize_features(features: pd.DataFrame) -> np.ndarray:
    """Z-score 标准化（按列），使各维度可比

    Args:
        features: 特征 DataFrame（不含 date 列）

    Returns:
        (n_days, n_features) 标准化后的 numpy 数组
    """
    feat_cols = [f[0] for f in FEATURE_DIMS]
    available_cols = [c for c in feat_cols if c in features.columns]

    if not available_cols:
        return np.zeros((len(features), 1))

    matrix = features[available_cols].values.astype(np.float32)

    # Z-score（加 epsilon 避免除零）
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    matrix = (matrix - mean) / std
    return matrix


# ============================================================================
# 缓存管理
# ============================================================================

def get_or_build_cache(code: str, df: Optional[pd.DataFrame] = None,
                       fetcher=None, cache_dir: str = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """获取或构建特征缓存

    流程：
    1. 缓存存在 → 增量更新
    2. 缓存不存在 + 传入了 df → 直接用 df 构建
    3. 缓存不存在 + 传入了 fetcher → 拉 3 年历史构建

    Args:
        code: 股票代码
        df: 可选，已准备好的 DataFrame（含所有指标）
        fetcher: 可选，DataFetcher 实例
        cache_dir: 缓存目录

    Returns:
        特征 DataFrame（含 date 列 + 15 维特征）
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{code}_features.parquet")

    # ---- 缓存存在：增量更新 ----
    if os.path.exists(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            if df is not None and len(df) > 0:
                # 检查是否有新数据
                if "date" in cached.columns and "date" in df.columns:
                    cached["date"] = pd.to_datetime(cached["date"])
                    df_dates = pd.to_datetime(df["date"])
                    last_cached_date = cached["date"].max()

                    new_data = df[df_dates > last_cached_date]
                    if len(new_data) > 3:
                        new_features = compute_feature_matrix(new_data)
                        cached = pd.concat([cached, new_features], ignore_index=True)
                        cached = cached.drop_duplicates(subset="date", keep="last")
                        cached.sort_values("date", inplace=True)
                        cached.reset_index(drop=True, inplace=True)
                        cached.to_parquet(cache_path, index=False)
                        logger.info(f"{code}: 特征缓存已更新 (+{len(new_features)}行)")

            return cached
        except Exception as e:
            logger.warning(f"{code}: 读取缓存失败 ({e})，将重建")

    # ---- 缓存不存在：构建 ----
    # 优先使用 fetcher 拉完整历史（如果可用），df 太小时不够做相似度匹配
    if fetcher is not None:
        logger.info(f"{code}: 首次拉取历史数据构建特征缓存 ({INITIAL_YEARS}年)")
        try:
            from datetime import datetime, timedelta
            start = (datetime.now() - timedelta(days=365 * INITIAL_YEARS)).strftime("%Y-%m-%d")
            hist_df = fetcher.daily_bars(code, start=start, end="")

            if not hist_df.empty and len(hist_df) >= 120:
                from src.data.indicators import add_all_indicators
                from src.data.cleaning import DataCleaner
                cleaner = DataCleaner()
                hist_df = cleaner.clean_single(hist_df, code)
                hist_df = add_all_indicators(hist_df)

                # 尝试富化（非必须）
                try:
                    from src.data.enrichment import enrich_all
                    idx_code = fetcher.market_index_for_stock(code)
                    index_df = fetcher.index_daily(idx_code, start=start)
                    fund_flow_df = fetcher.fund_flow(code)
                except Exception:
                    index_df = None
                    fund_flow_df = None
                try:
                    hist_df = enrich_all(hist_df, code, index_df=index_df, fund_flow_df=fund_flow_df)
                except Exception:
                    pass

                features = compute_feature_matrix(hist_df)
                features.to_parquet(cache_path, index=False)
                logger.info(f"{code}: 特征缓存已创建 ({len(features)}行)")
                return features
        except Exception as e:
            logger.error(f"{code}: 构建特征缓存失败: {e}，回退到 df 构建")

    # 回退：从已有 df 构建（至少需要 60 行）
    if df is not None and len(df) >= 60:
        logger.info(f"{code}: 从已有数据构建特征缓存 ({len(df)}行)")
        features = compute_feature_matrix(df)
        features.to_parquet(cache_path, index=False)
        return features

    logger.warning(f"{code}: 无数据源，无法构建特征缓存")
    return pd.DataFrame()


# ============================================================================
# 相似度搜索
# ============================================================================

def find_similar_days(features_df: pd.DataFrame,
                      top_k: int = 10,
                      exclude_recent: int = EXCLUDE_RECENT,
                      min_similarity: float = MIN_SIMILARITY) -> List[Tuple[int, float]]:
    """在特征矩阵中搜索与最新一天最相似的历史日

    Args:
        features_df: 特征 DataFrame（含 date 列 + 15维特征）
        top_k: 返回前 K 个最相似
        exclude_recent: 排除最近 N 天
        min_similarity: 最低相似度阈值

    Returns:
        [(index, similarity_score), ...] 按相似度降序
    """
    if features_df.empty or len(features_df) < exclude_recent + top_k:
        return []

    # 提取特征矩阵
    feat_cols = [f[0] for f in FEATURE_DIMS]
    available = [c for c in feat_cols if c in features_df.columns]
    if not available:
        return []

    matrix = features_df[available].values.astype(np.float32)

    # Z-score 标准化
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    matrix = (matrix - mean) / std

    # 当前特征向量 = 最后一天
    current_vec = matrix[-1]
    search_matrix = matrix[:-exclude_recent]

    if len(search_matrix) == 0:
        return []

    # 余弦相似度
    dot = np.dot(search_matrix, current_vec)
    norm_a = np.linalg.norm(search_matrix, axis=1)
    norm_b = np.linalg.norm(current_vec)
    similarities = dot / (norm_a * norm_b + 1e-10)

    # Top-K
    top_indices = np.argsort(similarities)[-top_k * 2:][::-1]  # 多取一些，后面再过滤

    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim < min_similarity:
            continue
        results.append((int(idx), sim))
        if len(results) >= top_k:
            break

    return results


# ============================================================================
# 统计预测推导
# ============================================================================

def derive_statistical_prediction(df: pd.DataFrame,
                                  similar_matches: List[Tuple[int, float]]) -> Dict:
    """从历史相似天的后续走势推导当前预测

    Args:
        df: 原始 OHLCV DataFrame（含 date/close，用于取后续实际涨跌）
        similar_matches: [(index_in_features_df, similarity), ...]
                         注意 index 对应的是 features_df 的行号，
                         需要同样定位到 df

    Returns:
        {
            "short_term": {days, direction, probability, change_pct, reason, ...},
            "mid_term":   {days, direction, probability, change_pct, reason, ...},
            "mid_long_term": {days, direction, probability, change_pct, reason, ...},
            "long_term":  {days, direction, probability, change_pct, reason, ...},
        }
    """
    short_samples = []
    mid_samples = []
    mid_long_samples = []
    long_samples = []

    for feat_idx, sim in similar_matches:
        if feat_idx < len(df):
            base_price = float(df["close"].iloc[feat_idx])

            # 短期: 5 天后的收益率
            if feat_idx + 5 < len(df):
                ret_5d = (float(df["close"].iloc[feat_idx + 5]) / base_price - 1) * 100
                short_samples.append((ret_5d, sim))

            # 中期: 15 天后的收益率
            if feat_idx + 15 < len(df):
                ret_15d = (float(df["close"].iloc[feat_idx + 15]) / base_price - 1) * 100
                mid_samples.append((ret_15d, sim))

            # 中长期: 40 天后的收益率
            if feat_idx + 40 < len(df):
                ret_40d = (float(df["close"].iloc[feat_idx + 40]) / base_price - 1) * 100
                mid_long_samples.append((ret_40d, sim))

            # 长期: 80 天后的收益率
            if feat_idx + 80 < len(df):
                ret_80d = (float(df["close"].iloc[feat_idx + 80]) / base_price - 1) * 100
                long_samples.append((ret_80d, sim))

    return {
        "short_term": _analyze_samples(short_samples, "3-5天"),
        "mid_term": _analyze_samples(mid_samples, "5-15天"),
        "mid_long_term": _analyze_samples(mid_long_samples, "15-40天"),
        "long_term": _analyze_samples(long_samples, "40-80天"),
    }


def _analyze_samples(samples: List[Tuple[float, float]], days_label: str) -> Dict:
    """对一组 (收益率%, 相似度) 样本做统计分析 → 预测结论

    Args:
        samples: [(return%, similarity), ...]
        days_label: "3-5天" 或 "10-20天"

    Returns:
        与 LLM 输出格式一致的预测字典
    """
    if not samples or len(samples) < MIN_SAMPLES:
        return {
            "days": days_label,
            "direction": "震荡",
            "probability": "小概率",
            "change_pct": 0.0,
            "reason": f"历史相似样本不足（{len(samples)}个 < {MIN_SAMPLES}个），无法给出可靠预测",
            "_sample_count": len(samples),
            "_warning": "sample_insufficient",
        }

    returns = np.array([s[0] for s in samples])
    similarities = np.array([s[1] for s in samples])

    # 加权统计
    weights = similarities / (similarities.sum() + 1e-10)
    weighted_mean = np.average(returns, weights=weights)
    weighted_up_ratio = float(np.sum(weights[returns > 0]))

    # 一致性：1 - 归一化标准差
    std_returns = np.std(returns)
    consistency = float(1.0 - min(std_returns / (abs(weighted_mean) + 1e-8), 1.0))

    # 加权中位数
    weighted_median = _weighted_percentile(returns, weights, 0.5)

    # ---- 方向判断 (基于加权上涨比例) ----
    if weighted_up_ratio >= 0.65:
        direction = "涨"
        probability = "大概率"
    elif weighted_up_ratio >= 0.50:
        direction = "涨"
        probability = "中等概率"
    elif weighted_up_ratio >= 0.35:
        direction = "震荡"
        probability = "中等概率"
    elif weighted_up_ratio >= 0.20:
        direction = "跌"
        probability = "中等概率"
    else:
        direction = "跌"
        probability = "大概率"

    # 如果一致性太低 + 上涨比例在 35-65% → 强行"震荡"
    if consistency < 0.3 and 0.30 < weighted_up_ratio < 0.70:
        direction = "震荡"
        probability = "中等概率"

    # ---- 理由 ----
    up_pct = int(weighted_up_ratio * 100)
    consistency_label = "高" if consistency > 0.6 else ("一般" if consistency > 0.3 else "低")
    reason = (
        f"历史{len(samples)}个相似样本中{up_pct}%上涨"
        f"（一致性{consistency_label}）"
    )

    return {
        "days": days_label,
        "direction": direction,
        "probability": probability,
        "change_pct": round(float(weighted_median), 1),
        "reason": reason,
        "_sample_count": len(samples),
        "_consistency": round(consistency, 2),
        "_weighted_up_ratio": round(weighted_up_ratio, 2),
        "_weighted_mean_return": round(float(weighted_mean), 1),
    }


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """加权分位数"""
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumsum = np.cumsum(sorted_weights)
    idx = np.searchsorted(cumsum, percentile * cumsum[-1])
    if idx >= len(sorted_values):
        return float(sorted_values[-1])
    return float(sorted_values[idx])


# ============================================================================
# 一站式接口（供 local_fusion 调用）
# ============================================================================

def predict_from_patterns(df: pd.DataFrame, code: str,
                          fetcher=None,
                          cache_dir: str = DEFAULT_CACHE_DIR) -> Dict:
    """一站式的统计预测接口

    自动处理缓存构建 → 相似度搜索 → 预测推导

    Args:
        df: 当日分析用的 DataFrame（含技术指标）
        code: 股票代码
        fetcher: DataFetcher 实例（首次构建缓存时需要）
        cache_dir: 缓存目录

    Returns:
        {"short_term": {...}, "mid_term": {...}, "mid_long_term": {...}, "long_term": {...}} 预测字典
    """
    # 1. 获取或更新特征缓存
    features_df = get_or_build_cache(code, df=df, fetcher=fetcher, cache_dir=cache_dir)

    _neutral_4way = {
        "short_term": {"direction": "震荡", "probability": "小概率",
                       "change_pct": 0.0, "reason": "无法构建特征缓存", "days": "3-5天"},
        "mid_term": {"direction": "震荡", "probability": "小概率",
                     "change_pct": 0.0, "reason": "无法构建特征缓存", "days": "5-15天"},
        "mid_long_term": {"direction": "震荡", "probability": "小概率",
                          "change_pct": 0.0, "reason": "无法构建特征缓存", "days": "15-40天"},
        "long_term": {"direction": "震荡", "probability": "小概率",
                      "change_pct": 0.0, "reason": "无法构建特征缓存", "days": "40-80天"},
    }

    if features_df.empty:
        return _neutral_4way

    # 2. 搜索相似天
    matches = find_similar_days(features_df)

    if not matches:
        no_match = dict(_neutral_4way)
        for k in no_match:
            no_match[k]["reason"] = "未找到足够相似的历史模式"
        return no_match

    # 3. 推导预测（使用 features_df 而非 df，因为 features_df 有完整的历史 close 列）
    predictions = derive_statistical_prediction(features_df, matches)

    # 附上相似度信息
    top_sim = matches[0][1] if matches else 0.0

    predictions["_meta"] = {
        "top_similarity": round(top_sim, 3),
        "match_count": len(matches),
        "cache_rows": len(features_df),
    }

    return predictions
