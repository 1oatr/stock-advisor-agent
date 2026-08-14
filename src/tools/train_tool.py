"""tools/train_tool.py — RL 模型训练工具

数据获取（个股 + 大盘 + 资金流向）→ 清洗 → 增强 → 指标计算 → 委托训练。
"""

import os
import logging
from datetime import datetime, timedelta

from src.data.fetcher import DataFetcher
from src.data.indicators import add_all_indicators
from src.data.cleaning import DataCleaner
from src.data.enrichment import enrich_all

logger = logging.getLogger(__name__)


def train_model(code: str, timesteps: int = 200000, update: bool = False,
                model_dir: str = "models", extra_callbacks: list = None,
                cancel_event=None, config: dict = None) -> dict:
    """训练或增量更新 RL 模型

    数据管线: 个股日线 → 大盘指数 → 资金流向 → 清洗 → 外部增强 → 指标计算 → 训练

    生命周期规则（由 rl/train.py 的 train_single_stock 处理）：
    - 首次训练 → 2年数据全量训练
    - <15天 → 增量微调
    - 15~60天 → 全量重训
    - >60天 → 自动删除旧模型，全量重训

    Args:
        code: 股票代码
        timesteps: 训练步数
        update: (保留参数，不再使用——train_single_stock 自动判断)
        model_dir: 模型目录
        extra_callbacks: (可选) 额外 SB3 回调列表，用于 GUI 训练进度/取消
        cancel_event: (可选) 可调用对象，返回 True 表示用户已取消

    Returns:
        {"status": "ok"|"updated"|"already_fresh", "eval_metrics": {...},
         "warnings": [...], "data_sources": {...}}
    """
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    warnings = []
    data_sources = {}

    # ---- 1. 获取个股日线 ----
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    df = fetcher.daily_bars(code, start=start, end="")
    if df.empty:
        return {"status": "error", "code": code, "message": f"{code} 无数据"}
    data_sources["daily_bars"] = f"{len(df)} 行"

    # ---- 2. 获取大盘指数数据 ----
    index_df = None
    try:
        idx_code = fetcher.market_index_for_stock(code)
        index_df = fetcher.index_daily(idx_code, start=start)
        if index_df.empty:
            logger.warning(f"{code}: 大盘 {idx_code} 数据为空")
            warnings.append(f"大盘指数 {idx_code} 数据为空，相关特征用 0 填充")
            index_df = None
        else:
            data_sources["index"] = f"{idx_code} ({len(index_df)} 行)"
    except Exception as e:
        logger.warning(f"{code}: 获取大盘数据失败 ({e})，用 0 填充")
        warnings.append(f"大盘指数获取失败 ({str(e)[:60]})，相关特征用 0 填充")

    # ---- 3. 获取资金流向 ----
    fund_flow_df = None
    try:
        fund_flow_df = fetcher.fund_flow(code)
        if fund_flow_df.empty:
            logger.warning(f"{code}: 资金流向数据为空")
            warnings.append("资金流向数据为空，相关特征用 0 填充")
            fund_flow_df = None
        else:
            data_sources["fund_flow"] = f"{len(fund_flow_df)} 行"
    except Exception as e:
        logger.warning(f"{code}: 获取资金流向失败 ({e})，用 0 填充")
        warnings.append(f"资金流向获取失败 ({str(e)[:60]})，相关特征用 0 填充")

    # ---- 4. 清洗 ----
    df = cleaner.clean_single(df, code)

    # ---- 5. 外部数据增强（大盘 + 资金 + 换手率） ----
    df = enrich_all(df, code, index_df=index_df, fund_flow_df=fund_flow_df)

    # ---- 6. 技术指标 ----
    df = add_all_indicators(df)

    if len(df) < 120:
        return {"status": "error", "code": code,
                "message": f"{code} 数据不足（需至少120天，实际{len(df)}天）"}

    # ---- 7. 委托训练 ----
    from src.rl.train import train_single_stock
    result = train_single_stock(df, code, timesteps=timesteps, model_dir=model_dir,
                                extra_callbacks=extra_callbacks,
                                cancel_event=cancel_event,
                                config=config)
    result["warnings"] = warnings
    result["data_sources"] = data_sources
    return result
