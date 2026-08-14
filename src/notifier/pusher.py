"""signal/pusher.py — 信号推送

微信 / APP 弹窗推送买卖信号、仓位调整提醒。
"""

from typing import Dict, Optional, List


class SignalMessage:
    """信号消息"""
    def __init__(self, code: str, action: str, price: float, reason: str = "",
                 confidence: float = 0.0, position: float = 0.0):
        self.code = code
        self.action = action
        self.price = price
        self.reason = reason
        self.confidence = confidence
        self.position = position
        self.timestamp = None  # 实际使用时注入

    def format_text(self) -> str:
        """格式化为可读文本"""
        icons = {"buy": "🟢 买入", "sell": "🔴 卖出", "hold": "⚪ 持有"}
        action_text = icons.get(self.action, self.action)
        text = (
            f"【股票信号】{self.code}\n"
            f"操作: {action_text}\n"
            f"价格: {self.price:.2f}\n"
            f"置信度: {self.confidence:.0%}\n"
            f"建议仓位: {self.position:.0%}\n"
        )
        if self.reason:
            text += f"原因: {self.reason}\n"
        return text


class SignalPusher:
    """信号推送器（多通道）"""

    def __init__(self):
        self.channels = {}  # name -> push function

    def register_channel(self, name: str, push_fn):
        """注册推送通道"""
        self.channels[name] = push_fn

    def push(self, msg: SignalMessage, channels: Optional[List[str]] = None):
        """推送信号到指定通道"""
        targets = channels or list(self.channels.keys())
        for name in targets:
            if name in self.channels:
                try:
                    self.channels[name](msg)
                except Exception as e:
                    print(f"[Pusher] {name} 推送失败: {e}")

    def push_batch(self, messages: List[SignalMessage], channels: Optional[List[str]] = None):
        """批量推送"""
        for msg in messages:
            self.push(msg, channels)
