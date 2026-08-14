"""memory/interested_stocks.py — 长期记忆：用户感兴趣股票

自动记录用户查过的每只股票（持久化 JSON），支持按频次/时间排序查询。
股票名称通过全市场动态解析获取，无需硬编码映射。
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

_MEMORY_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MEMORY_FILE = _MEMORY_DIR / "interested_stocks.json"


class InterestedStocks:
    """用户感兴趣的股票池（长期记忆）

    用法:
        >>> mem = InterestedStocks()
        >>> mem.record("600519")                # 记录一次查询
        >>> mem.record("000858", "五粮液")      # 带名称记录
        >>> mem.top(10)                          # 前10只最常查的
    """

    def __init__(self, filepath: Optional[Path] = None):
        self._file = filepath or _MEMORY_FILE
        self._stocks: dict = {}  # {code: {name, first_searched, last_searched, count, total}}
        self._loaded = False

    # ------------------------------------------------------------------
    # 内部：加载 / 保存
    # ------------------------------------------------------------------

    def _load(self):
        if self._loaded:
            return
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    self._stocks = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._stocks = {}
        self._loaded = True

    def _save(self):
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(self._stocks, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def record(self, code: str, name: str = ""):
        """记录一次查询（自动累加计数）"""
        self._load()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        if code in self._stocks:
            entry = self._stocks[code]
            entry["last_searched"] = now
            entry["count"] = entry.get("count", 1) + 1
            if name and not entry.get("name"):
                entry["name"] = name
        else:
            self._stocks[code] = {
                "name": name or self._resolve_name(code),
                "first_searched": now,
                "last_searched": now,
                "count": 1,
            }

        self._save()

    def all(self) -> list[dict]:
        """返回全部股票，按最近查询时间降序"""
        self._load()
        result = []
        for code, info in self._stocks.items():
            result.append({
                "code": code,
                "name": info.get("name", ""),
                "first_searched": info.get("first_searched", "?"),
                "last_searched": info.get("last_searched", "?"),
                "count": info.get("count", 1),
            })
        result.sort(key=lambda x: x["last_searched"], reverse=True)
        return result

    def top(self, n: int = 10) -> list[dict]:
        """按查询次数降序返回 top N"""
        self._load()
        result = []
        for code, info in self._stocks.items():
            result.append({
                "code": code,
                "name": info.get("name", ""),
                "first_searched": info.get("first_searched", "?"),
                "last_searched": info.get("last_searched", "?"),
                "count": info.get("count", 1),
            })
        # 主排：查询次数降序；次排：最近一次查询时间降序（稳定排序两遍）
        result.sort(key=lambda x: x["last_searched"], reverse=True)
        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:n]

    def get(self, code: str) -> Optional[dict]:
        """获取单只股票记录"""
        self._load()
        info = self._stocks.get(code)
        if info is None:
            return None
        return {
            "code": code,
            "name": info.get("name", ""),
            "first_searched": info.get("first_searched", "?"),
            "last_searched": info.get("last_searched", "?"),
            "count": info.get("count", 1),
        }

    def count(self) -> int:
        self._load()
        return len(self._stocks)

    # ------------------------------------------------------------------
    # 名称解析
    # ------------------------------------------------------------------

    def _resolve_name(self, code: str) -> str:
        """解析股票代码 → 名称（全A股动态查询）"""
        from src.data.name_resolver import resolve_stock_name
        return resolve_stock_name(code)


# ----------------------------------------------------------------------
# 单例
# ----------------------------------------------------------------------
_global_instance: Optional[InterestedStocks] = None


def get_interested_stocks() -> InterestedStocks:
    """获取 InterestedStocks 全局单例"""
    global _global_instance
    if _global_instance is None:
        _global_instance = InterestedStocks()
        _global_instance._load()
    return _global_instance


def record_stock_lookup(code: str, name: str = ""):
    """便捷函数：记录一次股票查询"""
    get_interested_stocks().record(code, name)
