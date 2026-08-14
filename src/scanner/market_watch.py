"""scanner/market_watch.py — 大盘环境判断与板块热点跟踪
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np


class MarketWatch:
    """大盘环境判断器

    判断当前市场状态：牛市 / 熊市 / 震荡市
    跟踪板块热点轮动
    """

    def __init__(self, fetcher=None):
        self.market_state: str = "unknown"
        self.hot_sectors: list = []
        self.fetcher = fetcher
        self._index_data: Dict[str, pd.DataFrame] = {}

    def update(self, fetcher=None) -> Dict:
        """更新大盘状态

        使用上证指数(000001)和深证成指(399001)判断市场状态。

        Returns:
            {"state": "bull", "index_ma_trend": "up", "sectors": [...]}
        """
        f = fetcher or self.fetcher
        if f is None:
            return {"state": self.market_state, "index_ma_trend": "unknown", "sectors": []}

        try:
            # 获取主要指数数据
            sh = f.index_bars("000001", start="")
            if sh.empty:
                return {"state": "unknown", "index_ma_trend": "unknown", "sectors": []}

            close = sh["close"]

            # 计算均线
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean() if len(close) > 60 else close

            last_close = close.iloc[-1]
            last_ma20 = ma20.iloc[-1]
            last_ma60 = ma60.iloc[-1] if len(close) > 60 else last_close

            # 趋势判断
            if len(close) > 20:
                ret_20d = (last_close / close.iloc[-20] - 1) * 100
                ret_60d = (last_close / close.iloc[-min(60, len(close))] - 1) * 100 if len(close) > 60 else 0
            else:
                ret_20d = 0
                ret_60d = 0

            # 牛/熊/震荡判断
            if last_close > last_ma20 > last_ma60 and ret_20d > 3:
                self.market_state = "bull"
                trend = "up"
            elif last_close < last_ma20 < last_ma60 and ret_20d < -3:
                self.market_state = "bear"
                trend = "down"
            else:
                self.market_state = "range"
                trend = "sideways"

            self._index_data["000001"] = sh

            # 获取热门板块
            try:
                if hasattr(f, 'sector_list') and hasattr(f, 'sector_flow'):
                    sectors_df = f.sector_list()
                    if not sectors_df.empty:
                        sector_names = sectors_df["板块名称"].head(10).tolist() if "板块名称" in sectors_df.columns else []
                        self.hot_sectors = sector_names[:5]
            except Exception:
                self.hot_sectors = []

            return {
                "state": self.market_state,
                "index_ma_trend": trend,
                "index_close": float(last_close),
                "ret_20d": round(ret_20d, 2),
                "ret_60d": round(ret_60d, 2),
                "sectors": self.hot_sectors,
            }

        except Exception as e:
            return {"state": "unknown", "index_ma_trend": "unknown", "error": str(e)}

    def get_market_state(self) -> str:
        """获取当前市场状态"""
        return self.market_state

    def get_hot_sectors(self, top_n: int = 5) -> list:
        """获取热门板块列表"""
        if not self.hot_sectors and self.fetcher:
            self.update()
        return self.hot_sectors[:top_n]

    def is_bull_market(self) -> bool:
        """是否为牛市环境"""
        return self.market_state == "bull"

    def is_bear_market(self) -> bool:
        """是否为熊市环境"""
        return self.market_state == "bear"

    def suggest_position_level(self) -> float:
        """根据大盘环境建议仓位水平

        Returns:
            建议仓位比例 0.0 ~ 1.0
        """
        levels = {
            "bull": 0.8,
            "range": 0.5,
            "bear": 0.2,
            "unknown": 0.3,
        }
        return levels.get(self.market_state, 0.3)
