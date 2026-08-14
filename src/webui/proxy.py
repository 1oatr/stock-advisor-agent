"""webui/proxy.py — 全局禁用系统代理

东财接口（实时快照/资金流向/个股信息）在 Windows 系统代理（Clash 类 VPN）下
请求会失败（ProxyError）。akshare 内部用 requests，会读取系统代理设置。

必须在 import 任何 akshare/requests 之前调用 disable_proxy_globally()。
仅作用于 WebUI 进程，不影响 CLI 与测试。
"""

import os


def disable_proxy_globally():
    """禁用所有系统代理，并 patch requests 强制直连。"""
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)

    # 兜底 patch requests：
    #   1) 未显式指定 proxies 时强制 {"http": None, "https": None}（东财接口依赖）
    #   2) 对东财域名强制 4s 快速失败（东财不可用时避免每次拖 14s 超时拖慢 WebUI）
    #      新浪等其它域名不受影响（保持默认长超时）
    try:
        import requests

        _orig = requests.sessions.Session.request
        EM_DOMAINS = ("eastmoney.com",)

        def _request(self, method, url, **kwargs):
            if kwargs.get("proxies") is None:
                kwargs["proxies"] = {"http": None, "https": None}
            url_lower = str(url).lower()
            if kwargs.get("timeout") is None and any(d in url_lower for d in EM_DOMAINS):
                kwargs["timeout"] = 4
            return _orig(self, method, url, **kwargs)

        requests.sessions.Session.request = _request
    except ImportError:
        pass
