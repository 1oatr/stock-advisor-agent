"""signal/monitor.py — 7×24 实时监控

定时扫描全市场标的，实时触发策略买卖信号。
"""

from typing import List, Dict, Optional, Callable
from datetime import datetime
import time
import threading
import pandas as pd


class MarketMonitor:
    """7×24 市场监控器

    定时轮询市场数据，运行策略并触发信号。
    """

    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable] = []  # 信号触发回调
        self.watchlist: List[str] = []
        self.last_signals: Dict[str, dict] = {}

    def start(self):
        """启动监控"""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def add_watch(self, code: str):
        """添加监控标的"""
        if code not in self.watchlist:
            self.watchlist.append(code)

    def remove_watch(self, code: str):
        """移除监控标的"""
        if code in self.watchlist:
            self.watchlist.remove(code)

    def register_callback(self, fn: Callable):
        """注册信号回调"""
        self.callbacks.append(fn)

    def _loop(self):
        """主循环"""
        while self.running:
            try:
                self._scan_once()
            except Exception as e:
                print(f"[Monitor] 扫描异常: {e}")
            time.sleep(self.interval)

    def _scan_once(self):
        """单次扫描"""
        if not self.watchlist:
            return

        # 获取实时行情
        quotes = self._fetch_quotes()

        # 运行策略判断
        for code in self.watchlist:
            if code in quotes:
                signal = self._evaluate(code, quotes[code])
                if signal and signal != self.last_signals.get(code):
                    self.last_signals[code] = signal
                    self._trigger_callbacks(code, signal)

    def _fetch_quotes(self) -> Dict[str, dict]:
        """获取行情数据"""
        # TODO: 接入实时数据
        return {}

    def _evaluate(self, code: str, quote: dict) -> Optional[dict]:
        """评估是否触发信号"""
        # TODO: 调用策略引擎 + 决策融合
        pass

    def _trigger_callbacks(self, code: str, signal: dict):
        """触发回调"""
        for cb in self.callbacks:
            try:
                cb(code, signal)
            except Exception as e:
                print(f"[Monitor] 回调异常: {e}")

    @property
    def is_running(self) -> bool:
        return self.running
