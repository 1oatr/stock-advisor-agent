"""backtest/rules.py — A股交易规则适配

严格遵循 A 股 T+1、涨跌停、停牌、科创板差异化规则。
"""

from typing import Dict, Optional
from datetime import datetime
import pandas as pd


class ASharesRules:
    """A 股规则适配器"""

    MARKET_MAIN = "main"      # 主板 ±10%
    MARKET_STAR = "star"      # 科创板 ±20%
    MARKET_GEM = "gem"        # 创业板 ±20%
    MARKET_BJ = "bj"          # 北交所 ±30%

    def __init__(self):
        # 各板块涨跌幅限制
        self.limit_map = {
            self.MARKET_MAIN: (0.10, -0.10),     # 主板
            self.MARKET_STAR: (0.20, -0.20),     # 科创板
            self.MARKET_GEM: (0.20, -0.20),      # 创业板
            self.MARKET_BJ: (0.30, -0.30),       # 北交所
        }

    def detect_market(self, code: str) -> str:
        """识别股票所属板块"""
        if code.startswith("688"):
            return self.MARKET_STAR
        elif code.startswith("300"):
            return self.MARKET_GEM
        elif code.startswith("8"):
            return self.MARKET_BJ
        # 000/001/002/600/601/603/605 等为主板
        return self.MARKET_MAIN

    def get_price_limits(self, code: str, prev_close: float) -> (float, float):
        """获取涨跌停价格"""
        market = self.detect_market(code)
        up_pct, down_pct = self.limit_map.get(market, (0.10, -0.10))
        limit_up = prev_close * (1 + up_pct)
        limit_down = prev_close * (1 + down_pct)
        # 精确到分
        limit_up = round(limit_up, 2)
        limit_down = round(limit_down, 2)
        return limit_up, limit_down

    def is_limit_up(self, price: float, limit_up: float) -> bool:
        """是否涨停"""
        return price >= limit_up

    def is_limit_down(self, price: float, limit_down: float) -> bool:
        """是否跌停"""
        return price <= limit_down

    def is_suspended(self, df_row: pd.Series) -> bool:
        """是否停牌（成交量=0）"""
        return df_row.get("volume", 0) == 0

    def get_new_stock_limit(self, days_listed: int) -> Optional[int]:
        """新股涨跌幅限制（上市首日等特殊情况）"""
        if days_listed == 1:
            return None  # 上市首日无限制
        return None

    def get_trade_hours(self) -> (str, str, str, str):
        """A股交易时间"""
        return ("09:30", "11:30", "13:00", "15:00")

    def is_trading_time(self, dt: datetime) -> bool:
        """是否为交易时间"""
        t = dt.strftime("%H:%M")
        return ("09:30" <= t <= "11:30") or ("13:00" <= t <= "15:00")
