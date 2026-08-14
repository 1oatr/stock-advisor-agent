"""tools — 工具封装层

将各分析模块封装为 LLM Agent 可调用的标准工具。
每个工具提供 JSON Schema 参数定义，供 DeepSeek 自主调度。
"""

from .registry import ToolDef, ToolRegistry, register_all_tools
from .analyze_tool import analyze_stock
from .llm_skills_tool import skills_analyze, llm_analyze
from .scan_tool import scan_market
from .predict_tool import predict_stocks
from .backtest_tool import run_backtest
from .train_tool import train_model
from .compare_tool import compare_stocks

__all__ = [
    "ToolDef", "ToolRegistry", "register_all_tools",
    "analyze_stock", "skills_analyze", "llm_analyze", "scan_market",
    "predict_stocks", "run_backtest", "train_model", "compare_stocks",
]
