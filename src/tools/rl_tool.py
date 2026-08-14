"""tools/rl_tool.py — RL 单股预测工具

轻量级工具，加载已训练的 RL 模型对单只股票做推理。
未训练时返回 untrained=True，由上游融合层处理并询问用户是否训练。
"""

import pandas as pd
from src.data.fetcher import DataFetcher
from src.data.indicators import add_all_indicators
from src.data.cleaning import DataCleaner


def get_rl_prediction(code: str, window: int = 60) -> dict:
    """获取 RL 模型对单只股票的买卖预测

    Args:
        code: 股票代码（6位数字）

    Returns:
        {
            "code": "000977",
            "action": "buy"|"sell"|"hold",
            "confidence": 0.65,
            "untrained": False,
            "model_fresh": True,
            "details": {...} 或 None
        }
    """
    try:
        from src.rl.agent import SingleStockAgent
        from src.data.enrichment import enrich_all

        fetcher = DataFetcher()

        # 1. 个股日线
        df = fetcher.daily_bars(code, start="", end="")
        if df.empty:
            return {"code": code, "action": "hold", "confidence": 0.5, "untrained": True, "reason": f"{code} 无数据"}

        # 2. 大盘指数
        index_df = None
        try:
            idx_code = fetcher.market_index_for_stock(code)
            index_df = fetcher.index_daily(idx_code)
        except Exception:
            pass

        # 3. 资金流向
        fund_flow_df = None
        try:
            fund_flow_df = fetcher.fund_flow(code)
        except Exception:
            pass

        # 4. 清洗 + 增强 + 指标
        df = DataCleaner().clean_single(df, code)
        df = enrich_all(df, code, index_df=index_df, fund_flow_df=fund_flow_df)
        df = add_all_indicators(df)
        if df.empty or len(df) < 30:
            return {"code": code, "action": "hold", "confidence": 0.5, "untrained": True, "reason": "数据不足"}

        agent = SingleStockAgent(window=window)
        result = agent.predict(df.tail(120), code)
        result["code"] = code
        return result

    except ImportError:
        return {"code": code, "action": "hold", "confidence": 0.5, "untrained": True, "reason": "RL 模块不可用"}
    except Exception as e:
        return {"code": code, "action": "hold", "confidence": 0.5, "untrained": True, "reason": str(e)[:80]}
