"""knowledge/explainer.py — LLM 解释器

使用 DeepSeek V4 Flash 生成自然语言的分析解释。
需设置环境变量 DEEPSEEK_API_KEY。
"""

from typing import Optional, Dict, List
import os
from openai import OpenAI


class LLMExplainer:
    """LLM 解释器

    将技术分析结果转化为可读的自然语言解释。
    使用 DeepSeek V4 Flash 模型（兼容 OpenAI API 格式）。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        enabled: bool = False,
    ):
        self.enabled = enabled
        self.client = None

        if enabled:
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY 未设置。请设置环境变量或在配置文件中提供。"
                )
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.model = model

    def generate_stock_report(
        self,
        code: str,
        stock_name: str,
        rl_decision: dict,
        rule_results: dict,
        market_state: str,
        skill_results: Optional[list] = None,
    ) -> str:
        """生成单只股票的完整分析报告

        Args:
            code: 股票代码
            stock_name: 股票名称
            rl_decision: RL 模型决策结果
            rule_results: 规则分析结果
            market_state: 当前市场状态
            skill_results: 技能分析结果列表

        Returns:
            自然语言分析报告
        """
        if not self.enabled or self.client is None:
            return self._fallback_report(rl_decision, rule_results, skill_results)

        prompt = self._build_prompt(code, stock_name, rl_decision, rule_results, market_state, skill_results)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._fallback_report(rl_decision, rule_results, skill_results) + f"\n[LLM调用异常: {e}]"

    def generate_multi_stock_summary(
        self, rankings: List[Dict], market_state: str
    ) -> str:
        """生成多股综合总结"""
        if not self.enabled or self.client is None:
            return self._fallback_summary(rankings)

        prompt = self._build_multi_prompt(rankings, market_state)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._fallback_summary(rankings) + f"\n[LLM调用异常: {e}]"

    def _build_prompt(self, code, name, rl_dec, rule_res, market, skill_results=None) -> str:
        """构建发给 LLM 的提示词"""
        skill_text = ""
        if skill_results:
            skill_lines = []
            for sr in skill_results:
                if hasattr(sr, 'explanation'):
                    skill_lines.append(f"    - {sr.skill_name}: {sr.signal} (置信度{sr.confidence:.0%})")
            if skill_lines:
                skill_text = "\n【决策技能分析】\n" + "\n".join(skill_lines)

        return f"""你是一位专业的股票分析顾问。请根据以下数据，对股票 {code}({name}) 给出综合分析报告。

【市场环境】{market}
【RL模型决策】动作={rl_dec.get('action', 'hold')}, 置信度={rl_dec.get('confidence', 0.5):.0%}
【技术规则分析】综合信号={rule_res.get('composite_signal', 'hold')}, 强度={rule_res.get('composite_strength', 0.5):.0%}, 共识度={rule_res.get('consensus', 0.5):.0%}
规则详情:
{chr(10).join(f"  - {r['name']}: {r['explanation']}" for r in rule_res.get('rules', []))}{skill_text}

请给出：1. 综合判断  2. 风险提示  3. 操作建议（包括仓位比例）"""

    def _build_multi_prompt(self, rankings: list, market: str) -> str:
        """构建多股总结提示词"""
        stock_lines = []
        for r in rankings[:10]:
            stock_lines.append(f"  {r.get('code', '')} → {r.get('action', 'hold')} (评分: {r.get('fusion_score', 0):.2f})")
        stocks_text = "\n".join(stock_lines)

        return f"""你是一位专业的股票分析顾问。请根据以下多股分析结果，给出综合投资建议。

【市场环境】{market}
【股票排名】
{stocks_text}

请给出：1. 重点关注方向  2. 板块分析  3. 仓位建议  4. 风险提示"""

    def _fallback_report(self, rl_decision: dict, rule_results: dict, skill_results: Optional[list] = None) -> str:
        """LLM 不可用时的回退报告"""
        signal = rule_results.get("composite_signal", "hold")
        strength = rule_results.get("composite_strength", 0.5)
        consensus = rule_results.get("consensus", 0.5)

        signal_map = {"buy": "买入", "sell": "卖出", "hold": "持有"}
        confidence_map = {"buy": "偏多", "sell": "偏空", "hold": "中性"}

        lines = [
            f"[规则引擎分析] 建议: {signal_map.get(signal, '持有')}",
            f"  信号强度: {strength:.0%}",
            f"  规则共识度: {consensus:.0%}",
            f"  市场观点: {confidence_map.get(signal, '中性')}",
        ]

        if skill_results:
            buy_count = sum(1 for s in skill_results if s.signal in ("buy", "strong_buy"))
            sell_count = sum(1 for s in skill_results if s.signal in ("sell", "strong_sell"))
            lines.append(f"  技能信号: {buy_count}个看多 / {sell_count}个看空")

        if rl_decision:
            rl_action = rl_decision.get("action", "hold")
            rl_conf = rl_decision.get("confidence", 0.5)
            lines.append(f"  RL模型: {signal_map.get(rl_action, '持有')} (置信度{rl_conf:.0%})")

        return "\n".join(lines)

    def _fallback_summary(self, rankings: list) -> str:
        """LLM 不可用时的回退摘要"""
        lines = ["[多股分析摘要]"]
        for i, r in enumerate(rankings[:5], 1):
            lines.append(f"  {i}. {r.get('code', '')} → {r.get('action', 'hold')} (评分: {r.get('fusion_score', 0):.2f})")
        return "\n".join(lines)
