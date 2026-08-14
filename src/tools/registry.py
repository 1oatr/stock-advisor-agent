"""tools/registry.py — 工具注册表

定义 ToolDef 标准接口和 ToolRegistry 全局注册表。
LLM Agent 通过注册表发现可用工具及其 JSON Schema。
"""

from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field


@dataclass
class ToolDef:
    """工具定义 — LLM 可调用的函数封装"""
    name: str                                    # 工具名，如 "analyze_stock"
    description: str                             # 一句话描述，告诉 LLM 这个工具干什么
    parameters: dict                             # JSON Schema 格式的参数定义
    function: Callable                           # 实际执行函数
    category: str = "analysis"                   # 分类：analysis / trading / training / market
    require_data: bool = True                    # 是否需要先拉数据


class ToolRegistry:
    """全局工具注册表

    用法:
        registry = ToolRegistry()
        registry.register(ToolDef(...))
        tools_schema = registry.to_openai_schema()   # 给 DeepSeek 的 function list
    """

    _tools: Dict[str, ToolDef] = {}

    @classmethod
    def register(cls, tool: ToolDef):
        """注册一个工具"""
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> ToolDef:
        """获取工具定义"""
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[ToolDef]:
        """列出所有工具"""
        return list(cls._tools.values())

    @classmethod
    def list_names(cls) -> List[str]:
        """列出所有工具名"""
        return list(cls._tools.keys())

    @classmethod
    def to_openai_schema(cls) -> List[dict]:
        """转为 OpenAI/DeepSeek function-calling 格式

        Returns:
            [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}, ...]
        """
        tools = []
        for tool in cls._tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return tools

    @classmethod
    def execute(cls, name: str, **kwargs) -> Any:
        """执行指定工具

        Args:
            name: 工具名
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        tool = cls._tools.get(name)
        if tool is None:
            return {"error": f"未知工具: {name}，可用: {list(cls._tools.keys())}"}
        try:
            return tool.function(**kwargs)
        except Exception as e:
            return {"error": f"工具 {name} 执行失败: {str(e)}"}


def register_all_tools():
    """注册所有内置工具（在各工具模块导入时自动调用）"""
    from .analyze_tool import analyze_stock
    from .llm_skills_tool import skills_analyze, llm_analyze
    from .scan_tool import scan_market
    from .predict_tool import predict_stocks
    from .backtest_tool import run_backtest
    from .train_tool import train_model
    from .compare_tool import compare_stocks
    from .rl_tool import get_rl_prediction

    tools = [
        ToolDef(
            name="analyze_stock",
            description="深度技术分析单只股票。输入股票代码，返回技术指标、规则引擎判断和技能分析结果。用于查看具体股票的走势、指标状态。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "股票代码，如 600519（贵州茅台）、000001（平安银行）"},
                    "days": {"type": "integer", "description": "回看天数，默认120天", "default": 120},
                },
                "required": ["code"],
            },
            function=analyze_stock,
            category="analysis",
        ),
        ToolDef(
            name="skills_analyze",
            description="本地策略引擎分析：运行11个技术技能 + 5维评分卡 + 规则交叉验证 + 统计预测。不依赖LLM/API，始终可用。输出与LLM分析相同格式的买卖决策。配合 analyze_stock 使用。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "股票代码"},
                    "days": {"type": "integer", "description": "回看天数", "default": 120},
                },
                "required": ["code"],
            },
            function=skills_analyze,
            category="analysis",
        ),
        ToolDef(
            name="llm_analyze",
            description="LLM深度解读（DeepSeek推理）：分析11个技能结果间的全局关联、信号矛盾与共识，输出独立决策 + 新闻舆情分析 + 价格预测。仅在LLM开启时可用。配合 skills_analyze 使用。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "股票代码"},
                    "days": {"type": "integer", "description": "回看天数", "default": 120},
                },
                "required": ["code"],
            },
            function=llm_analyze,
            category="analysis",
        ),
        ToolDef(
            name="scan_market",
            description="扫描全市场最活跃的股票。按涨跌幅、量比、均线形态等条件筛选并排名。用于发现热门标的。",
            parameters={
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "description": "返回前 N 只热门股", "default": 15},
                },
                "required": [],
            },
            function=scan_market,
            category="market",
        ),
        ToolDef(
            name="predict_stocks",
            description="对多只股票进行三方融合预测（规则引擎+RL智能体+LLM+Skills），输出综合买卖建议。最全面的股票预测入口。",
            parameters={
                "type": "object",
                "properties": {
                    "codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "股票代码列表，如 ['600519', '000858']",
                    },
                },
                "required": ["codes"],
            },
            function=predict_stocks,
            category="trading",
        ),
        ToolDef(
            name="run_backtest",
            description="对指定策略进行历史回测。用过去数据验证策略在真实A股规则下（T+1、涨跌停、手续费）的表现。",
            parameters={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "策略名: trend_following / mean_reversion / breakout / etf_grid",
                        "default": "trend_following",
                    },
                    "codes": {"type": "array", "items": {"type": "string"}, "description": "回测股票代码列表"},
                    "start": {"type": "string", "description": "起始日期 YYYY-MM-DD", "default": "2024-01-01"},
                    "end": {"type": "string", "description": "截止日期 YYYY-MM-DD", "default": "2025-06-30"},
                },
                "required": ["codes"],
            },
            function=run_backtest,
            category="trading",
        ),
        ToolDef(
            name="train_model",
            description="训练或更新某只股票的 RL 强化学习交易模型。模型学习历史买卖时机，输出买卖策略。训练后可参与 predict_stocks 的决策融合。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "股票代码"},
                    "timesteps": {"type": "integer", "description": "训练步数，默认200000", "default": 200000},
                    "update": {"type": "boolean", "description": "是否增量更新（15天内模型自动增量）", "default": False},
                },
                "required": ["code"],
            },
            function=train_model,
            category="training",
        ),
        ToolDef(
            name="compare_stocks",
            description="对比多只股票的技术面强弱。用于'茅台和五粮液谁更强'这类问题。",
            parameters={
                "type": "object",
                "properties": {
                    "codes": {"type": "array", "items": {"type": "string"}, "description": "要对比的股票代码列表"},
                },
                "required": ["codes"],
            },
            function=compare_stocks,
            category="analysis",
        ),
        ToolDef(
            name="get_rl_prediction",
            description="获取 RL 强化学习模型对单只股票的买卖预测。模型需先通过 train_model 训练。未训练时返回 untrained=True。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "股票代码"},
                },
                "required": ["code"],
            },
            function=get_rl_prediction,
            category="trading",
        ),
        ToolDef(
            name="get_market_state",
            description="获取当前大盘状态（牛/熊/震荡）和仓位建议。用于判断整体市场环境。",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            function=get_market_state,
            category="market",
        ),
    ]

    for tool in tools:
        ToolRegistry.register(tool)


def get_market_state() -> dict:
    """获取大盘状态"""
    from src.scanner.market_watch import MarketWatch
    from src.data.fetcher import DataFetcher

    fetcher = DataFetcher()
    market = MarketWatch(fetcher=fetcher)
    state = market.update()
    return {
        "market_state": state.get("state", "unknown"),
        "index_trend": state.get("index_ma_trend", "unknown"),
        "ret_20d": state.get("ret_20d", 0),
        "suggested_position": round(market.suggest_position_level() * 100),
        "hot_sectors": state.get("sectors", []),
    }
