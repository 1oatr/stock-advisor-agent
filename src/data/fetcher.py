"""data/fetcher.py — 多数据源接入

统一接口支持 Akshare / Tushare 合规数据源，可自定义扩展第三方接口。
"""

from typing import Optional, Dict, List
import pandas as pd


# ============================================================================
# 数据源基类
# ============================================================================

class DataSource:
    """数据源基类"""

    def __init__(self, name: str):
        self.name = name

    def daily_bars(self, code: str, start: str, end: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def realtime_quote(self, code: str) -> dict:
        raise NotImplementedError

    def index_bars(self, code: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError

    def fund_flow(self, code: str) -> pd.DataFrame:
        raise NotImplementedError


# ============================================================================
# Akshare 数据源
# ============================================================================

class AkshareSource(DataSource):
    """Akshare 数据源（无需 token，免费）"""

    def __init__(self):
        super().__init__("akshare")

    def daily_bars(self, code: str, start: str = "2018-01-01",
                   end: str = "2025-06-30", adjust: str = "qfq") -> pd.DataFrame:
        """获取A股个股日线行情

        Args:
            code: 股票代码，如 "000001" 或 "600519"
            start: 起始日期 "YYYY-MM-DD"（空字符串时获取全部数据）
            end: 截止日期 "YYYY-MM-DD"
            adjust: 复权 "qfq"(前复权) / "hfq"(后复权) / ""(不复权)

        Returns:
            DataFrame: date, open, high, low, close, volume, amount, code
        """
        import akshare as ak

        # 处理空字符串：使用 akshare 默认的最大范围
        start_fmt = start.replace("-", "") if start else "19900101"
        end_fmt = end.replace("-", "") if end else "21000118"

        # sz 或 sh 前缀
        market_code = ("sz" + code) if code.startswith(("0", "3")) else ("sh" + code)

        df = ak.stock_zh_a_daily(
            symbol=market_code,
            start_date=start_fmt,
            end_date=end_fmt,
            adjust=adjust,
        )

        if df.empty:
            return pd.DataFrame()

        # 标准化列名（保留换手率）
        col_map = {
            "date": "date", "open": "open", "close": "close",
            "high": "high", "low": "low", "volume": "volume",
        }
        # amount 可能来自不同列名
        for src, dst in [("amount", "amount"), ("turnover_rate", "turnover_rate"),
                         ("turnover", "turnover_rate"), ("换手率", "turnover_rate")]:
            if src in df.columns:
                col_map[src] = dst
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = code
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def realtime_quote(self, code: str) -> dict:
        """获取个股实时行情

        Returns:
            {"code": ..., "name": ..., "price": ..., "change": ..., "pct": ..., ...}
        """
        import akshare as ak

        # 使用 stock_zh_a_daily 获取最新行情日线作为近似
        market_code = ("sz" + code) if code.startswith(("0", "3")) else ("sh" + code)
        df = ak.stock_zh_a_daily(symbol=market_code, adjust="qfq")
        if df.empty:
            return {}

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        change = last["close"] - prev["close"]
        pct = change / prev["close"] * 100 if prev["close"] > 0 else 0

        return {
            "code": code,
            "name": code,
            "price": float(last["close"]),
            "change": float(change),
            "pct": float(pct),
            "volume": float(last["volume"]),
            "amount": float(last["amount"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "open": float(last["open"]),
        }

    def index_bars(self, code: str = "000001", start: str = "",
                   end: str = "") -> pd.DataFrame:
        """获取指数行情

        Args:
            code: 指数代码，"000001"(上证) "399001"(深证) "000688"(科创50)
        """
        import akshare as ak

        # stock_zh_index_daily 使用 sh/sz 前缀
        index_prefix = {"000001": "sh", "399001": "sz", "399006": "sz",
                        "000688": "sh", "000300": "sh", "000016": "sh", "000905": "sh"}
        prefix = index_prefix.get(code, "sh")
        symbol = f"{prefix}{code}"

        df = ak.stock_zh_index_daily(symbol=symbol)
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "date": "date", "open": "open", "close": "close",
            "high": "high", "low": "low",
            "volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = code

        # 过滤日期范围
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]

        df = df.sort_values("date").reset_index(drop=True)
        return df

    def fund_flow(self, code: str) -> pd.DataFrame:
        """获取个股资金流向"""
        import akshare as ak

        market = "sz" if code.startswith("0") or code.startswith("3") else "sh"
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        return df

    def sector_list(self) -> pd.DataFrame:
        """获取行业板块列表"""
        import akshare as ak
        return ak.stock_board_industry_name_em()

    def sector_flow(self, sector: str) -> pd.DataFrame:
        """获取板块资金流向"""
        import akshare as ak
        return ak.stock_board_industry_fund_flow_em(symbol=sector)

    def minute_bars(self, code: str, period: str = "5") -> pd.DataFrame:
        """获取分钟级K线

        Args:
            period: "1" / "5" / "15" / "30" / "60" 分钟
        """
        import akshare as ak
        return ak.stock_zh_a_minute(symbol=code, period=period, adjust="qfq")

    def all_stocks(self) -> pd.DataFrame:
        """获取A股全部股票列表（新浪接口）"""
        import akshare as ak
        try:
            return ak.stock_zh_a_spot()
        except Exception:
            # 备选方案
            try:
                return ak.stock_info_a_code_name()
            except Exception:
                return pd.DataFrame()

    def hot_stocks_rank(self) -> pd.DataFrame:
        """获取热门股票排名（涨幅榜）"""
        import akshare as ak
        try:
            df = ak.stock_zh_a_spot()
            if not df.empty and "涨跌幅" in df.columns:
                df = df.sort_values("涨跌幅", ascending=False).head(50)
            return df
        except Exception:
            return pd.DataFrame()


# ============================================================================
# Tushare 数据源（备选）
# ============================================================================

class TushareSource(DataSource):
    """Tushare 数据源（需 token）"""

    def __init__(self, token: str = ""):
        super().__init__("tushare")
        self.token = token
        self._pro = None

    def _get_pro(self):
        if self._pro is None and self.token:
            import tushare as ts
            ts.set_token(self.token)
            self._pro = ts.pro_api()
        return self._pro

    def daily_bars(self, code: str, start: str, end: str, **kwargs) -> pd.DataFrame:
        pro = self._get_pro()
        if pro is None:
            return pd.DataFrame()
        df = pro.daily(ts_code=code, start_date=start.replace("-", ""),
                       end_date=end.replace("-", ""))
        return df


# ============================================================================
# 统一入口
# ============================================================================

class DataFetcher:
    """统一数据获取入口，自动路由到对应数据源"""

    SOURCES = {
        "akshare": AkshareSource,
        "tushare": TushareSource,
    }

    def __init__(self, source: str = "akshare", **kwargs):
        if source not in self.SOURCES:
            raise ValueError(f"不支持的数据源: {source}，可选: {list(self.SOURCES.keys())}")
        self.source: DataSource = self.SOURCES[source](**kwargs)

    def daily_bars(self, code: str, start: str = "2018-01-01",
                   end: str = "2025-06-30", **kwargs) -> pd.DataFrame:
        """获取日线行情"""
        return self.source.daily_bars(code, start, end, **kwargs)

    def realtime_quote(self, code: str) -> dict:
        """获取实时行情"""
        return self.source.realtime_quote(code)

    def index_bars(self, code: str = "000001", start: str = "",
                   end: str = "") -> pd.DataFrame:
        """获取指数行情"""
        return self.source.index_bars(code, start, end)

    def batch_daily_bars(self, codes: List[str], start: str,
                         end: str) -> Dict[str, pd.DataFrame]:
        """批量获取多只股票日线数据"""
        return {code: self.daily_bars(code, start, end) for code in codes}

    def fund_flow(self, code: str) -> pd.DataFrame:
        """获取个股资金流向（每日主力/大单/小单净流入）"""
        return self.source.fund_flow(code)

    def index_daily(self, code: str = "000001", start: str = "",
                    end: str = "") -> pd.DataFrame:
        """获取指数日线行情

        Args:
            code: 000001(上证) 399001(深证) 399006(创业板) 000688(科创50)
        """
        return self.source.index_bars(code, start, end)

    def market_index_for_stock(self, stock_code: str) -> str:
        """根据个股代码返回对应大盘指数代码"""
        if stock_code.startswith("688"):
            return "000688"   # 科创50
        elif stock_code.startswith("3"):
            return "399006"   # 创业板指
        elif stock_code.startswith("0"):
            return "399001"   # 深证成指
        else:
            return "000001"   # 上证指数
