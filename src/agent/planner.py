"""agent/planner.py — DeepSeek 工具调度

把用户输入 + 可用工具列表发给 DeepSeek，
让 LLM 自主决定调用哪些工具、传什么参数。
"""

import os
import json
from typing import List, Dict, Optional


class Planner:
    """LLM 工具调度器

    接收用户自然语言 → DeepSeek 理解意图 → 返回 tool_call 列表
    """

    SYSTEM_PROMPT = """你是 A 股量化分析调度器。根据用户问题，返回必须调用的工具列表。

## 工具清单
- analyze_stock({code, days}): 单股技术分析
- skills_analyze({code, days}): 本地策略分析（**永远和 analyze_stock 成对出现，不依赖LLM**）
- llm_analyze({code, days}): LLM深度解读（**永远和 analyze_stock+skills_analyze 一起出现**）
- get_rl_prediction({code}): RL模型预测（**分析单股时永远一起调用**，不论模型是否已训练）
- compare_stocks({codes}): 多股对比排名
- predict_stocks({codes}): 三路融合预测
- scan_market({top_n}): 扫描热门股
- train_model({code, timesteps}): 训练RL模型
- run_backtest({strategy, codes, start, end}): 策略回测
- get_market_state({}): 大盘状态

## 代码映射
系统内置全A股名称→代码动态解析，无需硬编码映射。

## 调度规则（严格遵守，不得省略）
1. 分析单股("分析X"|"X怎么样"|"X走势"):
   → analyze_stock({code:X}) + skills_analyze({code:X}) + llm_analyze({code:X}) + get_rl_prediction({code:X})

2. 多股对比("对比X和Y"|"X和Y哪个好"|"X vs Y"):
   → analyze_stock({code:X}) + skills_analyze({code:X}) + llm_analyze({code:X}) + get_rl_prediction({code:X})
   + analyze_stock({code:Y}) + skills_analyze({code:Y}) + llm_analyze({code:Y}) + get_rl_prediction({code:Y})
   + compare_stocks({codes:[X,Y]})

3. 预测建议("预测"|"建议"|"买卖"):
   → predict_stocks({codes:[...]})

4. 扫描("扫描"|"热门"|"热点"):
   → scan_market({})

5. 训练("训练"|"学习"):
   → train_model({code:...})

6. 回测("回测"):
   → run_backtest({strategy:"trend_following", codes:[...], start:"2024-01-01", end:"2025-06-30"})

7. 打招呼(你好|谢谢|帮助):
   → [] (空数组)

只返回 tool_calls JSON 数组。"""

    def __init__(self, api_key: str = None, api_base: str = "https://api.deepseek.com",
                 model: str = "deepseek-v4-flash", llm_enabled: bool = True):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base
        self.model = model
        # LLM 关闭时（全局开关）plan() 不调用 DeepSeek，直接走关键词回退
        self.llm_enabled = llm_enabled
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        return self._client

    def plan(self, user_input: str, tools_schema: List[dict],
             context: str = "") -> List[dict]:
        """根据用户输入规划工具调用

        Args:
            user_input: 用户原始输入
            tools_schema: ToolRegistry.to_openai_schema() 的返回
            context: SessionMemory.get_context_for_llm() 的上下文

        Returns:
            [{"name": "analyze_stock", "params": {"code": "600519", "days": 120}}, ...]
        """
        # LLM 全局关闭 → 直接关键词回退，绝不调用 DeepSeek
        if not self.llm_enabled or self.client is None:
            calls = self._fallback_plan(user_input, tools_schema, context)
            # LLM 关闭时计划里不出现 llm_analyze（避免多余占位执行）
            if not self.llm_enabled:
                calls = [c for c in calls if c.get("name") != "llm_analyze"]
            return calls

        # 构建消息
        system = self.SYSTEM_PROMPT
        if context:
            system += f"\n\n## 会话上下文\n{context}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=600,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                calls = []
                for tc in msg.tool_calls:
                    params = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    calls.append({
                        "name": tc.function.name,
                        "params": params,
                    })
                return calls
            else:
                return []  # LLM 决定不需要工具

        except Exception as e:
            print(f"  [Planner] DeepSeek API 异常: {e}")
            return self._fallback_plan(user_input, tools_schema, context)

    def _fallback_plan(self, user_input: str, tools_schema: List[dict],
                       context: str = "") -> List[dict]:
        """LLM 不可用时的关键词匹配回退"""
        text = user_input.lower()

        # 训练/学习
        if any(w in text for w in ["训练", "学习", "train"]):
            code = self._extract_code(user_input)
            if code:
                return [{"name": "train_model", "params": {"code": code}}]

        # 扫描/热门
        if any(w in text for w in ["扫描", "热门", "热点", "盘点", "scan"]):
            return [{"name": "scan_market", "params": {"top_n": 15}}]

        # 预测/建议（必须在"对比"之前，否则"预测A和B"会被"和"误拦截）
        if any(w in text for w in ["预测", "建议", "买卖", "操作"]):
            codes = self._extract_codes(user_input)
            if codes:
                return [{"name": "predict_stocks", "params": {"codes": codes}}]

        # 对比/比较（"和"仅在检测到2+代码时才触发对比）
        compare_words = any(w in text for w in ["对比", "比较", "vs", "哪个"])
        codes = self._extract_codes(user_input)
        if compare_words and len(codes) >= 2:
            return [
                {"name": "analyze_stock", "params": {"code": codes[0]}},
                {"name": "skills_analyze", "params": {"code": codes[0]}},
                {"name": "llm_analyze", "params": {"code": codes[0]}},
                {"name": "get_rl_prediction", "params": {"code": codes[0]}},
                {"name": "analyze_stock", "params": {"code": codes[1]}},
                {"name": "skills_analyze", "params": {"code": codes[1]}},
                {"name": "llm_analyze", "params": {"code": codes[1]}},
                {"name": "get_rl_prediction", "params": {"code": codes[1]}},
                {"name": "compare_stocks", "params": {"codes": codes}},
            ]
        # "和" + 明确2只股票 → 也触发对比
        if "和" in text and len(codes) >= 2:
            return [
                {"name": "analyze_stock", "params": {"code": codes[0]}},
                {"name": "skills_analyze", "params": {"code": codes[0]}},
                {"name": "llm_analyze", "params": {"code": codes[0]}},
                {"name": "get_rl_prediction", "params": {"code": codes[0]}},
                {"name": "analyze_stock", "params": {"code": codes[1]}},
                {"name": "skills_analyze", "params": {"code": codes[1]}},
                {"name": "llm_analyze", "params": {"code": codes[1]}},
                {"name": "get_rl_prediction", "params": {"code": codes[1]}},
                {"name": "compare_stocks", "params": {"codes": codes}},
            ]

        # 默认：分析股票
        code = self._extract_code(user_input)
        if code:
            return [
                {"name": "analyze_stock", "params": {"code": code}},
                {"name": "skills_analyze", "params": {"code": code}},
                {"name": "llm_analyze", "params": {"code": code}},
                {"name": "get_rl_prediction", "params": {"code": code}},
            ]

        # 没有任何关键词 → 默认扫描
        return [{"name": "scan_market", "params": {"top_n": 10}}]

    # 常见简称→代码（这些简称不在 akshare 官方名称里，需单独维护）
    _SHORT_NAMES = {
        "茅台": "600519", "宁德": "300750", "平安": "601318",
        "招行": "600036", "格力": "000651", "美的": "000333",
        "伊利": "600887", "恒瑞": "600276", "海康": "002415",
        "隆基": "601012", "浪潮": "000977", "中兴": "000063",
        "顺丰": "002352", "牧原": "002714", "爱尔": "300015",
        "汇川": "300124", "紫金": "601899",
    }

    def _extract_code(self, text: str) -> Optional[str]:
        """从文本中提取股票代码（动态解析全A股 + 常见简称）"""
        import re
        # 1. 直接匹配6位数字（两侧只要非数字即可，允许"分析600519"这种中文紧贴）
        match = re.search(r'(?<!\d)\d{6}(?!\d)', text)
        if match:
            return match.group(0)

        # 2. 常见简称匹配（如"茅台"、"浪潮"）
        for short, code in self._SHORT_NAMES.items():
            if short in text:
                return code

        # 3. 全A股动态解析（名称→代码）
        from src.data.name_resolver import get_name_resolver
        resolver = get_name_resolver()
        # 尝试在文本中匹配已知股票名
        return resolver.find_code_in_text(text)

    def _extract_codes(self, text: str) -> List[str]:
        """从文本中提取多个股票代码（动态解析全A股 + 常见简称）"""
        import re
        codes = re.findall(r'(?<!\d)\d{6}(?!\d)', text)
        if codes:
            return list(dict.fromkeys(codes))  # 去重保序

        result = []
        # 1. 常见简称匹配
        for short, code in self._SHORT_NAMES.items():
            if short in text and code not in result:
                result.append(code)
        # 2. 全A股动态搜索
        from src.data.name_resolver import get_name_resolver
        resolver = get_name_resolver()
        for code in resolver.find_all_codes_in_text(text):
            if code not in result:
                result.append(code)
        return result
