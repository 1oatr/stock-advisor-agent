"""webui/market_catalog.py — 多市场代码模型与内置目录

统一代码格式:
    A股  裸 6 位数字          600519
    港股  hk: + 5 位数字       hk:00700
    美股  us: + ticker         us:AAPL

港股/美股的热门列表与名称解析依赖东财接口（当前网络不稳定），
因此内置常见股票池作为兜底。
"""

# ============================================================================
# 代码解析
# ============================================================================

def parse_code(code: str):
    """解析规范代码 → (market, symbol)

    market in ('a', 'hk', 'us')
    """
    code = (code or "").strip()
    if code.startswith("hk:"):
        return ("hk", code[3:].zfill(5))
    if code.startswith("us:"):
        return ("us", code[3:].upper())
    if len(code) == 6 and code.isdigit():
        return ("a", code)
    return ("a", code)  # 兜底按 A股


def to_canonical(market: str, symbol: str) -> str:
    """把 (market, symbol) 转回规范代码"""
    if market == "hk":
        return "hk:" + str(symbol).zfill(5)
    if market == "us":
        return "us:" + str(symbol).upper()
    return str(symbol).zfill(6) if symbol.isdigit() else str(symbol)


MARKET_NAMES = {"a": "A股", "hk": "港股", "us": "美股"}


def market_name(market: str) -> str:
    return MARKET_NAMES.get(market, market)


# ============================================================================
# 内置名称池（东财不可用时的名称兜底）
# ============================================================================

HK_NAME_MAP = {
    "00700": "腾讯控股", "09988": "阿里巴巴-W", "03690": "美团-W",
    "00005": "汇丰控股", "01299": "友邦保险", "00388": "香港交易所",
    "01810": "小米集团-W", "09618": "京东集团-SW", "01211": "比亚迪股份",
    "00941": "中国移动", "00981": "中芯国际", "00939": "建设银行",
    "01398": "工商银行", "02318": "中国平安", "01024": "快手-W",
    "02020": "安踏体育", "02331": "李宁", "06618": "京东健康",
    "01088": "中国神华", "00883": "中国海洋石油", "00939": "建设银行",
    "02628": "中国人寿", "00966": "中国太平", "01288": "农业银行",
    "00386": "中国石油化工股份",
}

US_NAME_MAP = {
    "AAPL": "苹果", "MSFT": "微软", "NVDA": "英伟达", "GOOGL": "谷歌-A",
    "AMZN": "亚马逊", "META": "Meta平台", "TSLA": "特斯拉", "BRK.B": "伯克希尔",
    "JPM": "摩根大通", "V": "Visa", "NFLX": "奈飞", "AMD": "超威半导体",
    "BABA": "阿里巴巴", "PDD": "拼多多", "INTC": "英特尔", "NKE": "耐克",
    "DIS": "迪士尼", "KO": "可口可乐", "PEP": "百事可乐", "WMT": "沃尔玛",
    "T": "AT&T", "XOM": "埃克森美孚", "CVX": "雪佛龙", "MCD": "麦当劳",
    "IBM": "IBM", "ORCL": "甲骨文", "CRM": "Salesforce", "UBER": "优步",
}

# A股兜底池（蓝筹，scanner 兜底用）
A_HOT_FALLBACK = [
    ("600519", "贵州茅台"), ("000858", "五粮液"), ("300750", "宁德时代"),
    ("601318", "中国平安"), ("600036", "招商银行"), ("000651", "格力电器"),
    ("000333", "美的集团"), ("600887", "伊利股份"), ("600276", "恒瑞医药"),
    ("002594", "比亚迪"), ("600030", "中信证券"), ("000001", "平安银行"),
    ("600900", "长江电力"), ("000977", "浪潮信息"), ("002230", "科大讯飞"),
]


def resolve_market_name(market: str, symbol: str) -> str:
    """市场感知名称解析，查不到返回 symbol 本身"""
    if market == "a":
        return symbol
    if market == "hk":
        return HK_NAME_MAP.get(symbol, symbol)
    if market == "us":
        return US_NAME_MAP.get(symbol.upper(), symbol)
    return symbol


def get_pool(market: str) -> list:
    """返回某市场的兜底股票池 [(code, name), ...]"""
    if market == "hk":
        return [(c, n) for c, n in HK_NAME_MAP.items()]
    if market == "us":
        return [(c, n) for c, n in US_NAME_MAP.items()]
    return A_HOT_FALLBACK
