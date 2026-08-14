"""agent/executor.py — 工具执行器

解析 Planner 返回的 tool_call 列表 → 调用 ToolRegistry → 收集结果
"""

from typing import List, Dict, Any
from src.tools.registry import ToolRegistry
from .memory import SessionMemory


class Executor:
    """工具执行器"""

    def __init__(self, memory: SessionMemory = None, llm_enabled: bool = True):
        self.memory = memory or SessionMemory()
        self.llm_enabled = llm_enabled

    def execute(self, tool_calls: List[dict]) -> list:
        """执行一组工具调用

        Args:
            tool_calls: Planner.plan() 的输出
                [{"name": "analyze_stock", "params": {"code": "600519"}}, ...]

        Returns:
            [(tool_name, result_dict), ...]  保持顺序，同名工具不互相覆盖
        """
        results = []

        for i, call in enumerate(tool_calls):
            name = call.get("name", "")
            params = call.get("params", {})

            tool = ToolRegistry.get(name)
            if tool is None:
                results.append((name, {"error": f"未知工具: {name}"}))
                continue

            # 为同名多次调用生成唯一标识
            unique_key = f"{name}[{i}]" if tool_calls.count(call) > 1 or any(
                c.get("name") == name for j, c in enumerate(tool_calls) if j != i
            ) else name

            print(f"  🔧 执行: {unique_key}({params})")

            # 当 LLM 关闭时，跳过 llm_analyze 调用，返回占位结果
            if name == "llm_analyze" and not self.llm_enabled:
                print(f"  ⏭️  跳过: {unique_key}（LLM已关闭，使用 skills_analyze 替代）")
                placeholder = {
                    "action": "hold",
                    "confidence": 0.5,
                    "disabled": True,
                    "analysis_text": "LLM 已关闭，未执行深度分析。参见 skills_analyze 的本地策略分析结果。",
                    "key_signals": [],
                    "risk_note": "LLM 未启用",
                    "predictions": {
                        "short_term": {"days": "3-5天", "direction": "震荡", "probability": "小概率",
                                       "change_pct": 0.0, "reason": "LLM 未启用"},
                        "mid_term": {"days": "5-15天", "direction": "震荡", "probability": "小概率",
                                     "change_pct": 0.0, "reason": "LLM 未启用"},
                        "mid_long_term": {"days": "15-40天", "direction": "震荡", "probability": "小概率",
                                          "change_pct": 0.0, "reason": "LLM 未启用"},
                        "long_term": {"days": "40-80天", "direction": "震荡", "probability": "小概率",
                                      "change_pct": 0.0, "reason": "LLM 未启用"},
                    },
                    "code": params.get("code", ""),
                    "source": "llm_disabled",
                }
                results.append((name, placeholder))
                continue

            # 当 LLM 关闭时，predict_stocks 三路融合强制走本地（use_llm=False）
            if name == "predict_stocks" and not self.llm_enabled:
                params = dict(params)
                params["use_llm"] = False
                print(f"  ⏭️  {unique_key}：LLM 已关闭，三路融合使用本地增强策略")

            try:
                result = tool.function(**params)
                results.append((name, result))
                self.memory.add_tool_call(name, params, result)
            except Exception as e:
                error_result = {"error": str(e), "params": params}
                results.append((name, error_result))
                self.memory.add_tool_call(name, params, error_result)
                print(f"  ❌ {name} 失败: {e}")

        return results

    def execute_single(self, name: str, **params) -> Any:
        """执行单个工具"""
        return self.execute([{"name": name, "params": params}]).get(name, {})
