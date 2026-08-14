"""data/cache.py — 数据缓存系统

本地+内存双缓存，提升回测与模型训练速度。
"""

from typing import Optional, Dict
import pandas as pd
import os
import pickle


class DataCache:
    """本地文件缓存"""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """读取缓存"""
        path = self._path(key)
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def set(self, key: str, df: pd.DataFrame):
        """写入缓存"""
        path = self._path(key)
        df.to_parquet(path, index=False)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def clear(self):
        """清空缓存"""
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))

    def _path(self, key: str) -> str:
        safe = key.replace("/", "_").replace(":", "_").replace(".", "_")
        return os.path.join(self.cache_dir, f"{safe}.parquet")
