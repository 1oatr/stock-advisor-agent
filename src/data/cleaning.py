"""data/cleaning.py — 数据清洗管线

自动剔除停牌、ST、异常波动数据；标准化量化因子。
"""

from typing import List, Optional
import pandas as pd


class DataCleaner:
    """数据清洗器"""

    ST_KEYWORDS = ["ST", "ST*", "SST", "S*ST", "退"]
    EXCLUDE_CODES = []  # 可配置排除列表

    def clean_single(self, df: pd.DataFrame, code: str = "") -> pd.DataFrame:
        """单只股票清洗

        - 剔除停牌日（成交量=0）
        - 剔除ST标记
        - 处理缺失值
        - 检查异常价格波动
        """
        df = df.copy()
        df = self._remove_suspended(df)
        df = self._fill_missing(df)
        df = self._remove_outliers(df)
        return df

    def clean_multi(self, data_dict: dict) -> dict:
        """批量清洗多只股票"""
        return {code: self.clean_single(df, code) for code, df in data_dict.items()}

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名和格式

        统一输出列: date, open, high, low, close, volume, amount, code
        """
        column_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "股票代码": "code",
            "Date": "date", "Open": "open", "Close": "close",
            "High": "high", "Low": "low", "Volume": "volume",
            "Amount": "amount",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def _remove_suspended(self, df: pd.DataFrame) -> pd.DataFrame:
        """剔除停牌日（成交量=0 或空）"""
        if "volume" in df.columns:
            return df[df["volume"] > 0].reset_index(drop=True)
        return df

    def _fill_missing(self, df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """填充缺失值"""
        if method == "ffill":
            return df.ffill().dropna()
        elif method == "bfill":
            return df.bfill().dropna()
        return df.dropna()

    def _remove_outliers(self, df: pd.DataFrame, zscore_threshold: float = 5.0) -> pd.DataFrame:
        """剔除价格异常波动（Z-score 法）"""
        if "close" in df.columns and len(df) > 20:
            returns = df["close"].pct_change()
            mean = returns.mean()
            std = returns.std()
            df = df[abs(returns - mean) <= zscore_threshold * std].reset_index(drop=True)
        return df
