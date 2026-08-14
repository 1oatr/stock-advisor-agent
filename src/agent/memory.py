"""agent/memory.py — 全会话记忆

记住本轮所有对话内容，供 LLM 后续推理时引用。
用户说"那和五粮液对比"时，LLM 能从记忆中知道上一轮在讨论茅台。
"""

from typing import List, Dict
from datetime import datetime


class SessionMemory:
    """全会话上下文记忆"""

    def __init__(self, max_history: int = 50):
        self.messages: List[Dict] = []          # LLM 对话历史
        self.tool_calls: List[Dict] = []         # 本轮工具调用记录
        self.analyzed_stocks: List[str] = []     # 本轮分析过的股票
        self.user_prefs: Dict = {}               # 用户偏好
        self.max_history = max_history

    def add_user_message(self, content: str):
        """记录用户消息"""
        self.messages.append({"role": "user", "content": content, "time": datetime.now().isoformat()})
        self._trim()

    def add_assistant_message(self, content: str):
        """记录 AI 回复"""
        self.messages.append({"role": "assistant", "content": content, "time": datetime.now().isoformat()})
        self._trim()

    def add_tool_call(self, tool_name: str, params: dict, result: dict):
        """记录一次工具调用"""
        self.tool_calls.append({
            "tool": tool_name,
            "params": params,
            "result_summary": self._summarize_result(result),
            "time": datetime.now().isoformat(),
        })
        # 自动追踪分析过的股票代码
        if "code" in params and params["code"] not in self.analyzed_stocks:
            self.analyzed_stocks.append(params["code"])
        if "codes" in params:
            for c in params["codes"]:
                if c not in self.analyzed_stocks:
                    self.analyzed_stocks.append(c)

    def get_context_for_llm(self) -> str:
        """生成发送给 LLM 的上下文摘要"""
        parts = []

        if self.analyzed_stocks:
            parts.append(f"本轮已分析股票: {', '.join(self.analyzed_stocks)}")

        if self.tool_calls:
            parts.append(f"本轮已调用 {len(self.tool_calls)} 次工具:")
            for tc in self.tool_calls[-5:]:  # 最近5次
                parts.append(f"  - {tc['tool']}({tc['params']}) → {tc['result_summary'][:80]}")

        if self.user_prefs:
            parts.append(f"用户偏好: {self.user_prefs}")

        return "\n".join(parts) if parts else ""

    def get_recent_context(self, n: int = 3) -> str:
        """获取最近 N 轮工具调用上下文"""
        recent = self.tool_calls[-n:]
        lines = []
        for tc in recent:
            lines.append(f"[{tc['time'][:19]}] {tc['tool']}: {tc['result_summary'][:100]}")
        return "\n".join(lines)

    def clear(self):
        """清空记忆"""
        self.messages.clear()
        self.tool_calls.clear()
        self.analyzed_stocks.clear()
        self.user_prefs.clear()

    def _summarize_result(self, result: dict) -> str:
        """压缩工具结果为短摘要"""
        if "error" in result:
            return f"错误: {result['error'][:80]}"
        if "hot_stocks" in result:
            return f"扫描到 {len(result['hot_stocks'])} 只股票"
        if "recommendations" in result:
            actions = [f"{r['code']}={r['fused']['action']}" for r in result.get("recommendations", [])]
            return f"预测: {', '.join(actions)}"
        if "code" in result:
            rule = result.get("rule_engine", {}).get("composite_signal", "?")
            return f"{result['code']} 规则={rule} 收盘={result.get('last_close', '?')}"
        if "comparison_table" in result:
            return f"对比 {result.get('stock_count', 0)} 只股票"
        return "工具执行完成"

    def _trim(self):
        """保持消息数不超过上限"""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
