"""tools/llm_skills_tool.py — 技能分析工具（拆分为 LLM + 非LLM 两个独立分析）

本模块包含两个独立工具：
- skills_analyze: 非LLM本地策略分析（LocalFusionEngine 5维评分卡），始终运行
- llm_analyze:    LLM深度分析（DeepSeek推理），仅在 /llm on 时运行
"""

import json
import os
import pandas as pd
from src.data.fetcher import DataFetcher
from src.data.indicators import add_all_indicators
from src.data.cleaning import DataCleaner
from src.skills import init_skills, get_manager


# ============================================================================
# 工具 1: skills_analyze — 非LLM本地策略分析（始终运行）
# ============================================================================

def skills_analyze(code: str, days: int = 500, market: str = "a",
                   df: pd.DataFrame = None) -> dict:
    """本地策略引擎分析：11技能 + 规则交叉验证 + 5维评分卡 + 统计预测

    不依赖 LLM/API，始终可用。
    输入股票代码，返回与 LLM 分析相同格式的决策结果。

    Args:
        code: 股票代码
        days: 回看天数
        market: 市场 'a'/'hk'/'us'
        df: 预取的日线 DataFrame（多市场时由 WebUI 注入）

    Returns:
        {action, confidence, analysis_text, key_signals, risk_note, predictions, source: "local_fusion"}
    """
    # 1. 获取数据
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    init_skills()
    skill_mgr = get_manager()

    if df is None:
        df = fetcher.daily_bars(code, start="", end="")
    if df.empty:
        return {"action": "hold", "confidence": 0.3, "error": f"无数据", "code": code}

    df = cleaner.clean_single(df, code)
    df = add_all_indicators(df)
    recent_df = df.tail(min(days, len(df)))

    # 记录查询（长期记忆）
    from src.memory import record_stock_lookup
    record_stock_lookup(code)

    # 1.5 技能配置：开关 / 启停 / 融合参数 / 置信度阈值
    from src.webui import services
    scfg = services.get_skills_config()
    if not scfg.get("enabled", True):
        return {"action": "hold", "confidence": 0.4, "disabled": True,
                "reason": "技能引擎已关闭", "code": code}
    from src.skills import SkillRegistry
    switches = scfg.get("skill_switches", {})
    active_names = [s.name for s in SkillRegistry.list_all() if switches.get(s.name, True)]

    # 2. 运行启用的技能
    skill_results = skill_mgr.run_selected(recent_df, active_names, code)

    # 3. 本地增强策略引擎
    from src.agent.local_fusion import LocalFusionEngine
    from src.knowledge.analyzer import RuleAnalyzer

    # 获取规则决策（用于交叉验证）；非 A 股禁涨跌停规则
    exclude_rules = ["limit_gap"] if market != "a" else None
    rule_analyzer = RuleAnalyzer()
    try:
        rule_decision_raw = rule_analyzer.analyze(recent_df, code, exclude_rules=exclude_rules)
        rule_decision = {
            "signal": rule_decision_raw.get("composite_signal", "hold"),
            "strength": rule_decision_raw.get("composite_strength", 0.3),
            "top_rules": rule_decision_raw.get("top_rules", []),
        }
    except Exception:
        rule_decision = {"signal": "hold", "strength": 0.3, "top_rules": []}

    # 大盘状态
    market_state = "range"
    try:
        from src.scanner.market_watch import MarketWatch
        mw = MarketWatch(fetcher=fetcher)
        ms = mw.update()
        market_state = ms.get("state", "range")
    except Exception:
        pass

    # 融合引擎参数 + 技能信号置信度阈值（低于阈值的 buy/sell 不参与打分）
    fusion_cfg = dict(scfg.get("fusion") or {})
    fusion_cfg["confidence_threshold"] = scfg.get("confidence_threshold", 0.0)
    engine = LocalFusionEngine(config=fusion_cfg)
    return engine.analyze(
        skill_results=skill_results,
        rule_decision=rule_decision,
        market_state=market_state,
        df=recent_df,
        code=code,
        fetcher=fetcher,
    )


# ============================================================================
# 工具 2: llm_analyze — LLM深度分析（仅在 /llm on 时运行）
# ============================================================================

def llm_analyze(code: str, days: int = 500, api_key: str = None,
                api_base: str = "https://api.deepseek.com",
                model: str = "deepseek-v4-flash", market: str = "a",
                df: pd.DataFrame = None) -> dict:
    """LLM 深度解读 11 个技能结果 + 技术指标，给出独立决策 + 价格预测

    仅在 LLM 开启时由 Executor 调度。LLM 关闭时 Executor 会跳过此工具。

    Args:
        code: 股票代码
        days: 回看天数
        api_key: DeepSeek API key（默认读取环境变量 DEEPSEEK_API_KEY）
        api_base: API 地址
        model: 模型名
        market: 市场 'a'/'hk'/'us'
        df: 预取的日线 DataFrame（多市场时由 WebUI 注入）

    Returns:
        {action, confidence, analysis_text, key_signals, risk_note, predictions, source: "llm_skills"}
    """
    # 未显式传入时用默认值（WebUI 可能传 None）
    if not api_base:
        api_base = "https://api.deepseek.com"
    if not model:
        model = "deepseek-v4-flash"

    # 1. 获取数据
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    init_skills()
    skill_mgr = get_manager()

    if df is None:
        df = fetcher.daily_bars(code, start="", end="")
    if df.empty:
        return {"action": "hold", "confidence": 0.3, "error": f"无数据", "code": code}

    df = cleaner.clean_single(df, code)
    df = add_all_indicators(df)
    recent_df = df.tail(min(days, len(df)))

    # 记录查询（长期记忆）
    from src.memory import record_stock_lookup
    record_stock_lookup(code)

    # 2. 运行启用的技能（与本地分析共用 skill_switches 配置）
    from src.webui import services
    scfg = services.get_skills_config()
    switches = scfg.get("skill_switches", {})
    from src.skills import SkillRegistry
    active_names = [s.name for s in SkillRegistry.list_all() if switches.get(s.name, True)]
    skill_results = skill_mgr.run_selected(recent_df, active_names, code)

    # 3. 收集指标快照
    last = recent_df.iloc[-1]
    indicators_text = f"""
收盘价: {last['close']:.2f}
MA5: {last.get('MA5', '?'):.2f}  MA20: {last.get('MA20', '?'):.2f}  MA60: {last.get('MA60', '?'):.2f}
RSI: {last.get('RSI', '?'):.1f}
MACD: DIF={last.get('MACD_DIF', '?'):.2f}  DEA={last.get('MACD_DEA', '?'):.2f}  HIST={last.get('MACD_HIST', '?'):.2f}
布林带: 上{last.get('BOLL_UP', '?'):.2f} 中{last.get('BOLL_MID', '?'):.2f} 下{last.get('BOLL_DN', '?'):.2f}  %B={last.get('BOLL_POSITION', '?'):.0%}
ATR%: {last.get('ATR_PCT', '?'):.2f}%  量比: {last.get('VOL_RATIO', '?'):.2f}
CCI: {last.get('CCI', '?'):.1f}  KDJ: K={last.get('KDJ_K', '?'):.1f} D={last.get('KDJ_D', '?'):.1f}
5日涨跌: {recent_df['close'].pct_change(5).iloc[-1]*100:.1f}%  20日涨跌: {recent_df['close'].pct_change(20).iloc[-1]*100:.1f}%
"""

    # 4. 收集技能分析结果
    skills_text_parts = []
    for sr in skill_results:
        skills_text_parts.append(
            f"[{sr.skill_name}] 信号={sr.signal} 置信度={sr.confidence:.0%} 强度={sr.strength:.0%}\n"
            f"   解释: {sr.explanation}\n"
            f"   检测到形态: {', '.join(sr.patterns_detected) if sr.patterns_detected else '无'}"
        )
    skills_text = "\n\n".join(skills_text_parts)

    # 4b. 提取新闻舆情数据（如果有）
    news_text = ""
    for sr in skill_results:
        if sr.skill_name == "news_sentiment" and sr.metadata:
            news_items = sr.metadata.get("news_items", [])
            if news_items:
                news_parts = []
                for n in news_items[:5]:
                    news_parts.append(f"· [{n['time'][:10]}] {n['title']}")
                news_text = "\n".join(news_parts)
            break

    skill_count = len(skill_results)

    # 5. 构建 prompt → DeepSeek
    system_prompt = f"""你是一位资深量化分析师，拥有 20 年 A 股交易经验。

你的任务：分析以下 {skill_count} 个技能的输出 + 技术指标数据 + 近期新闻，给出独立的买卖决策。

要求：
1. 综合评估所有信号，判断它们是否相互印证还是矛盾
2. 结合市场指标（RSI位置、布林带位置、量比等）判断信号可信度
3. **结合近期新闻判断消息面影响**：利好/利空/中性，是否会改变技术面判断
4. **必须给出短期和长期价格预测**：基于技术面+消息面，预估未来走势的方向和幅度
5. 输出严格 JSON 格式的决策结果
6. analysis_text 要详细写出你的分析思路，为什么看多/看空/持有
7. 操作建议力度说明（action 和 confidence 会根据 confidence 自动转换）：
   - confidence≥0.70 对应"强烈建议"，0.50-0.69 对应"建议"，<0.50 对应"略微考虑"
8. predictions 中提供 4 个时间维度的预测：
   - short_term: 短期（3-5个交易日）
   - mid_term: 中期（5-15个交易日）
   - mid_long_term: 中长期（15-40个交易日）
   - long_term: 长期（40-80个交易日）
   - direction: "涨" / "跌" / "震荡"
   - probability: "大概率"(>60%) / "中等概率"(40-60%) / "小概率"(<40%)
   - change_pct: 预估涨跌幅百分比，正数涨负数跌
   - reason: 一句话说明依据"""

    # 新闻板块
    news_section = ""
    if news_text:
        news_section = f"""
══════════════ 近期新闻舆情 ══════════════
{news_text}
"""

    user_prompt = f"""股票: {code} | 回看 {days} 天

══════════════ 技术指标快照 ══════════════
{indicators_text}
{news_section}
══════════════ {skill_count} 个技能分析结果 ══════════════
{skills_text}

══════════════ 请给出你的分析 ══════════════

请从以下角度分析：
1. 各技能信号之间存在多少共识？有无关键矛盾？
2. 新闻舆情偏向利好还是利空？力度如何？
3. 结合 RSI/布林带/成交量位置，当前处于什么状态（超买/超卖/趋势中）？
4. 最终操作建议是什么？把握度如何？

返回严格 JSON 格式（不要 markdown 代码块）：
{{"action": "buy|sell|hold", "confidence": 0.0-1.0, "analysis_text": "你的详细分析（200字内）", "key_signals": ["最重要的买入信号", "最重要的卖出信号"], "risk_note": "主要风险提示", "news_sentiment": "新闻舆情一句话总结", "predictions": {{"short_term": {{"days": "3-5天", "direction": "涨|跌|震荡", "probability": "大概率|中等概率|小概率", "change_pct": 数字, "reason": "一句话依据"}}, "mid_term": {{"days": "5-15天", "direction": "涨|跌|震荡", "probability": "大概率|中等概率|小概率", "change_pct": 数字, "reason": "一句话依据"}}, "mid_long_term": {{"days": "15-40天", "direction": "涨|跌|震荡", "probability": "大概率|中等概率|小概率", "change_pct": 数字, "reason": "一句话依据"}}, "long_term": {{"days": "40-80天", "direction": "涨|跌|震荡", "probability": "大概率|中等概率|小概率", "change_pct": 数字, "reason": "一句话依据"}}}}}}"""

    # 6. 调用 DeepSeek
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        # 无 API key 时用本地回退
        return _fallback_llm_skills(skill_results, code)

    # 全局 LLM 开关关闭 → 本地回退（拦截任何遗漏入口，绝不调 LLM API）
    from src.llm_config import llm_enabled as _global_llm_enabled
    if not _global_llm_enabled():
        fallback = _fallback_llm_skills(skill_results, code)
        fallback["disabled"] = True
        fallback["reason"] = "LLM 已关闭"
        return fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )

        content = response.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # JSON 截断修复：补全缺失的闭合括号
            result = _repair_truncated_json(content)

        result["code"] = code
        result["source"] = "llm_skills"
        return result

    except Exception as e:
        # 解析或 API 失败时回退
        fallback = _fallback_llm_skills(skill_results, code)
        fallback["llm_error"] = str(e)[:100]
        return fallback


# ============================================================================
# 辅助函数
# ============================================================================

def _repair_truncated_json(content: str) -> dict:
    """修复被截断的 JSON — 补全缺失的 } ] }}} """
    # 统计括号
    open_braces = content.count("{") - content.count("}")
    open_brackets = content.count("[") - content.count("]")

    # 检查是否在字符串中间截断
    in_string = False
    for i, ch in enumerate(content):
        if ch == '"' and (i == 0 or content[i-1] != '\\'):
            in_string = not in_string

    # 如果在字符串中间截断，去掉最后的不完整字符串
    if in_string:
        # 找到最后一个完整的键值对
        last_comma = content.rfind(',"')
        if last_comma > 0:
            content = content[:last_comma]
        else:
            content = content.rstrip() + '"'

    # 补全括号
    content = content.rstrip(",\n\r ")
    content += "]" * open_brackets
    content += "}" * open_braces

    return json.loads(content)


def _fallback_llm_skills(skill_results, code: str) -> dict:
    """LLM 不可用时的本地回退：加权汇总技能结果 + 统计预测"""
    from src.skills import init_skills, get_manager
    init_skills()
    mgr = get_manager()
    agg = mgr.aggregate_signal(skill_results)

    # 用加权方式模拟 LLM 推理
    buy_count = len(agg.get("buy_skills", []))
    sell_count = len(agg.get("sell_skills", []))

    if agg["signal"] == "buy" and buy_count >= sell_count * 2:
        analysis = f"多技能共振看多：{buy_count}个技能给出买入信号，信号一致性高。"
    elif agg["signal"] == "sell" and sell_count >= buy_count * 2:
        analysis = f"多技能共振看空：{sell_count}个技能给出卖出信号，信号一致性高。"
    elif buy_count == sell_count:
        analysis = "多空信号均衡，无明显方向，建议持有。"
    elif agg["signal"] == "buy":
        analysis = f"偏多但信号分歧（{buy_count}买 vs {sell_count}卖），需谨慎。"
    else:
        analysis = f"偏空但信号分歧（{buy_count}买 vs {sell_count}卖），需确认。"

    # 基于信号强度的统计预测
    strength = agg.get("strength", 0.5)
    signal = agg["signal"]
    confidence = round(agg["confidence"], 2)

    predictions = {
        "short_term": {
            "days": "3-5天",
            "direction": "涨" if signal == "buy" else ("跌" if signal == "sell" else "震荡"),
            "probability": "大概率" if confidence > 0.6 else ("中等概率" if confidence > 0.4 else "小概率"),
            "change_pct": round(strength * 3 * (1 if signal == "buy" else -1), 1),
            "reason": f"{buy_count}个买入信号 vs {sell_count}个卖出信号" if signal != "hold" else "多空均衡",
        },
        "mid_term": {
            "days": "5-15天",
            "direction": "涨" if signal == "buy" else ("跌" if signal == "sell" else "震荡"),
            "probability": "大概率" if confidence > 0.55 else ("中等概率" if confidence > 0.35 else "小概率"),
            "change_pct": round(strength * 5 * (1 if signal == "buy" else -1), 1),
            "reason": f"中期趋势跟随短期方向" if signal != "hold" else "信号不明朗",
        },
        "mid_long_term": {
            "days": "15-40天",
            "direction": "涨" if signal == "buy" else ("跌" if signal == "sell" else "震荡"),
            "probability": "中等概率" if confidence > 0.4 else "小概率",
            "change_pct": round(strength * 8 * (1 if signal == "buy" else -1), 1),
            "reason": f"中长期延续性预估" if signal != "hold" else "方向不明朗",
        },
        "long_term": {
            "days": "40-80天",
            "direction": "涨" if signal == "buy" else ("跌" if signal == "sell" else "震荡"),
            "probability": "中等概率" if confidence > 0.35 else "小概率",
            "change_pct": round(strength * 12 * (1 if signal == "buy" else -1), 1),
            "reason": f"长期趋势预判，把握度较低" if signal != "hold" else "方向不明朗",
        },
    }

    return {
        "action": agg["signal"],
        "confidence": confidence,
        "analysis_text": analysis,
        "key_signals": agg.get("buy_skills", [])[:3] + agg.get("sell_skills", [])[:3],
        "risk_note": "LLM 不可用，使用本地加权回退",
        "predictions": predictions,
        "code": code,
        "source": "fallback_skills",
    }
