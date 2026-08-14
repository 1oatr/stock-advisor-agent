"""signal/logger.py — 交易日志

完整记录每一笔信号触发时间、标的、价格、仓位、触发条件。
"""

from typing import List, Dict, Optional
from datetime import datetime
import json
import os
import pandas as pd


class TradeLogger:
    """交易日志记录器"""

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records: List[dict] = []

    def log_signal(self, code: str, action: str, price: float, source: str,
                   confidence: float = 0.0, position: float = 0.0,
                   reason: str = "", trigger_rules: Optional[list] = None):
        """记录信号触发"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": "signal",
            "code": code,
            "action": action,
            "price": price,
            "source": source,          # "rl" / "rules" / "fusion"
            "confidence": confidence,
            "position": position,
            "reason": reason,
            "trigger_rules": trigger_rules or [],
        }
        self.records.append(record)
        self._append_to_file(record)

    def log_trade(self, code: str, direction: str, price: float, volume: int,
                  cost: float, pnl: float = 0.0, balance: float = 0.0):
        """记录成交"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": "trade",
            "code": code,
            "direction": direction,
            "price": price,
            "volume": volume,
            "cost": cost,
            "pnl": pnl,
            "balance": balance,
        }
        self.records.append(record)
        self._append_to_file(record)

    def log_risk(self, code: str, rule: str, reason: str, approved: bool):
        """记录风控拦截"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": "risk",
            "code": code,
            "rule": rule,
            "reason": reason,
            "approved": approved,
        }
        self.records.append(record)
        self._append_to_file(record)

    def get_today_logs(self) -> List[dict]:
        """获取当日日志"""
        today = datetime.now().strftime("%Y-%m-%d")
        return [r for r in self.records if r["timestamp"].startswith(today)]

    def export_to_dataframe(self) -> pd.DataFrame:
        """导出为 DataFrame"""
        return pd.DataFrame(self.records)

    def _append_to_file(self, record: dict):
        """追加写入日志文件"""
        date_str = datetime.now().strftime("%Y%m%d")
        path = os.path.join(self.log_dir, f"trades_{date_str}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
