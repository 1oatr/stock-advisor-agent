"""data/name_resolver.py — 股票代码 ↔ 名称动态解析器

从 akshare 获取全市场 5000+ 只股票列表，建立代码↔名称的双向映射。
结果缓存在本地 JSON 文件，每天自动刷新一次。

替代之前散落在各处的硬编码 STOCK_NAME_MAP。
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_FILE = _CACHE_DIR / "stock_names_cache.json"
_CACHE_TTL_HOURS = 24  # 股票名称变化极少，24小时刷新足够

# 兜底映射（网络完全不可用时用）
_FALLBACK_MAP: dict[str, str] = {
    "600519": "贵州茅台", "000858": "五粮液", "300750": "宁德时代",
    "601318": "中国平安", "600036": "招商银行", "000651": "格力电器",
    "000333": "美的集团", "600887": "伊利股份", "600276": "恒瑞医药",
    "002415": "海康威视", "002594": "比亚迪", "601012": "隆基绿能",
    "600030": "中信证券", "000001": "平安银行", "000002": "万科A",
    "601166": "兴业银行", "600900": "长江电力", "000568": "泸州老窖",
    "002714": "牧原股份", "300059": "东方财富", "000977": "浪潮信息",
    "000725": "京东方A", "002475": "立讯精密", "000063": "中兴通讯",
    "002352": "顺丰控股", "601899": "紫金矿业", "300274": "阳光电源",
    "002230": "科大讯飞", "300015": "爱尔眼科", "300124": "汇川技术",
}


class StockNameResolver:
    """股票名称 ↔ 代码双向解析器（本地缓存 + 自动刷新）"""

    def __init__(self):
        self._code_to_name: dict[str, str] = {}
        self._name_to_code: dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        """去掉股票名中的空格（akshare 数据常见"五 粮 液"→"五粮液"）"""
        return name.replace(" ", "").replace("　", "")

    def resolve_code(self, name_or_code: str) -> Optional[str]:
        """输入名称或代码，返回纯数字代码。已确认是代码则直接返回。

        >>> resolver.resolve_code("五粮液")  → "000858" (自动匹配"五 粮 液")
        >>> resolver.resolve_code("600519")  → "600519"
        """
        self._ensure_loaded()

        # 已经是纯数字代码 → 直接返回
        if name_or_code.isdigit() and len(name_or_code) == 6:
            return name_or_code

        query = self._normalize(name_or_code)

        # 精确匹配（去空格后）
        if query in self._name_to_code:
            return self._name_to_code[query]

        # 模糊匹配：用户输入短名缩略（如"茅台"匹配"贵州茅台"）
        for name, code in self._name_to_code.items():
            if query in name:
                return code

        return None

    def resolve_name(self, code: str) -> str:
        """输入代码，返回名称。

        >>> resolver.resolve_name("600519") → "贵州茅台"
        """
        self._ensure_loaded()
        return self._code_to_name.get(code, "")

    def find_code_in_text(self, text: str) -> Optional[str]:
        """在任意文本中查找股票名并返回代码。

        优先匹配最长名称，同时兼容 akshare 数据中的空格（如"五 粮 液"→"五粮液"）。

        >>> resolver.find_code_in_text("分析浪潮信息最近走势") → "000977"
        >>> resolver.find_code_in_text("对比五粮液和茅台") → "000858"
        """
        self._ensure_loaded()
        clean_text = self._normalize(text)
        best_name = ""
        best_code = None

        for name, code in self._name_to_code.items():
            if len(name) > len(best_name) and name in clean_text:
                best_name = name
                best_code = code

        return best_code

    def find_all_codes_in_text(self, text: str) -> list[str]:
        """在文本中查找所有匹配的股票代码（用于多股对比等场景）"""
        self._ensure_loaded()
        clean_text = self._normalize(text)
        found = []
        for name, code in self._name_to_code.items():
            if name in clean_text and code not in found:
                found.append(code)
        return found

    def search(self, keyword: str, limit: int = 10) -> list[tuple[str, str]]:
        """模糊搜索股票名称，返回 [(code, name), ...]"""
        self._ensure_loaded()
        results = []
        for name, code in self._name_to_code.items():
            if keyword in name:
                results.append((code, name))
            if len(results) >= limit:
                break
        return results

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._code_to_name)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return

        # 1. 尝试从缓存加载
        if self._load_cache():
            self._loaded = True
            return

        # 2. 从 akshare 在线获取
        if self._fetch_online():
            self._loaded = True
            return

        # 3. 兜底：用内置小 map
        self._load_fallback()
        self._loaded = True

    def _load_cache(self) -> bool:
        """加载本地缓存，如果过期则返回 False"""
        if not _CACHE_FILE.exists():
            return False

        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            updated_str = data.get("updated", "")
            if updated_str:
                updated = datetime.fromisoformat(updated_str)
                if datetime.now() - updated > timedelta(hours=_CACHE_TTL_HOURS):
                    return False  # 过期，触发在线刷新

            self._code_to_name = data.get("code_to_name", {})
            # 增量构建 name→code
            self._name_to_code = {v: k for k, v in self._code_to_name.items()}
            return bool(self._code_to_name)

        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    def _fetch_online(self) -> bool:
        """从 akshare 在线获取全市场股票列表"""
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            if df.empty:
                return False

            self._code_to_name = {}
            for _, row in df.iterrows():
                code = str(row.get("code", "")).strip()
                name = self._normalize(str(row.get("name", "")))
                if code and len(code) == 6 and name:
                    self._code_to_name[code] = name

            self._name_to_code = {v: k for k, v in self._code_to_name.items()}
            self._save_cache()
            return bool(self._code_to_name)

        except Exception as e:
            logger.warning(f"在线获取股票列表失败: {e}")
            return False

    def _save_cache(self):
        """保存到本地 JSON 文件"""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "updated": datetime.now().isoformat(),
                    "count": len(self._code_to_name),
                    "code_to_name": self._code_to_name,
                }, f, ensure_ascii=False)
        except IOError as e:
            logger.warning(f"保存股票名缓存失败: {e}")

    def _load_fallback(self):
        """加载硬编码兜底映射"""
        self._code_to_name = dict(_FALLBACK_MAP)
        self._name_to_code = {v: k for k, v in _FALLBACK_MAP.items()}


# ----------------------------------------------------------------------
# 全局单例
# ----------------------------------------------------------------------
_resolver: Optional[StockNameResolver] = None


def get_name_resolver() -> StockNameResolver:
    global _resolver
    if _resolver is None:
        _resolver = StockNameResolver()
    return _resolver


def resolve_stock_code(name_or_code: str) -> Optional[str]:
    """便捷函数：输入中文名或数字代码，返回纯数字代码。"""
    return get_name_resolver().resolve_code(name_or_code)


def resolve_stock_name(code: str) -> str:
    """便捷函数：输入数字代码，返回中文名。"""
    return get_name_resolver().resolve_name(code)
