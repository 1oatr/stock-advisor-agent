"""tools/predict_tool.py — 三路融合预测工具

规则引擎 + RL智能体 + LLM+Skills → 加权融合 → 综合买卖建议
"""

import pandas as pd
from src.data.fetcher import DataFetcher
from src.data.indicators import add_all_indicators
from src.data.cleaning import DataCleaner
from src.knowledge.analyzer import RuleAnalyzer
from src.skills import init_skills, get_manager
from src.advisor.fuser import DecisionFuser
from src.advisor.ranker import StockRanker
from .analyze_tool import analyze_stock
from .llm_skills_tool import skills_analyze, llm_analyze


def predict_stocks(codes: list, api_key: str = None, use_llm: bool = True,
                   api_base: str = None, model: str = None) -> dict:
    """三路融合预测多只股票

    Args:
        codes: 股票代码列表
        api_key: DeepSeek API key
        use_llm: 是否使用 LLM（False=使用本地增强策略）
        api_base: LLM API 地址（None=用 llm_analyze 默认）
        model: LLM 模型名（None=用 llm_analyze 默认）

    Returns:
        {
            "market_state": "bear",
            "recommendations": [
                {
                    "code": "600519",
                    "llm_skills": {"action": "buy", "confidence": 0.72, "reasoning": "..."},
                    "rules": {"signal": "sell", "strength": 0.42, "top_signals": [...]},
                    "rl": {"action": "hold", "confidence": 0.50},
                    "fused": {"action": "buy", "confidence": 0.51, "position": 0.12,
                              "consensus": "low", "divergence_note": "RL与规则分歧"}
                },
                ...
            ],
            "ranking": [按推荐力度排序],
        }
    """
    # 全局 LLM 开关关闭 → 强制本地增强策略，任何入口都不调 LLM
    from src.llm_config import llm_enabled as _global_llm_enabled
    if use_llm and not _global_llm_enabled():
        use_llm = False

    from src.scanner.market_watch import MarketWatch

    fetcher = DataFetcher()
    market = MarketWatch(fetcher=fetcher)
    market_state = market.update()
    market_label = market_state.get("state", "range")

    # 初始化
    init_skills()
    skill_mgr = get_manager()
    fuser = DecisionFuser()
    ranker = StockRanker()
    analyzer = RuleAnalyzer()

    recommendations = []
    fused_list = []

    for code in codes:
        # 1. 规则引擎（复用 analyze_stock 的结果）
        analysis = analyze_stock(code)
        if "error" in analysis:
            recommendations.append({"code": code, "error": analysis["error"]})
            continue

        rule_decision = {
            "signal": analysis["rule_engine"]["composite_signal"],
            "strength": analysis["rule_engine"]["composite_strength"],
            "consensus": analysis["rule_engine"]["consensus"],
        }

        # 2. 本地策略分析（始终运行）+ LLM 深度分析（仅在 LLM ON 时）
        skills_decision = skills_analyze(code)
        if use_llm:
            llm_decision = llm_analyze(code, api_key=api_key, api_base=api_base, model=model)
        else:
            llm_decision = None  # LLM 关闭时不调用，skills_decision 独立展示

        # 融合用 LLM 结果（有则用，无则回退到本地策略）
        fusion_llm = llm_decision if llm_decision else skills_decision

        # 3. RL 决策（尝试加载模型）
        rl_decision = {"action": "hold", "confidence": 0.5}
        try:
            from src.rl.agent import SingleStockAgent
            from src.data.enrichment import enrich_all
            agent = SingleStockAgent()
            df = fetcher.daily_bars(code, start="", end="")
            if not df.empty:
                # 外部数据（失败时用 0 填充，不影响主流程）
                index_df = fund_flow_df = None
                try:
                    idx_code = fetcher.market_index_for_stock(code)
                    index_df = fetcher.index_daily(idx_code)
                except Exception:
                    pass
                try:
                    fund_flow_df = fetcher.fund_flow(code)
                except Exception:
                    pass
                df = DataCleaner().clean_single(df, code)
                df = enrich_all(df, code, index_df=index_df, fund_flow_df=fund_flow_df)
                df = add_all_indicators(df)
                rl_decision = agent.predict(df.tail(120), code)
        except Exception:
            pass  # 模型未训练或维度不匹配，用默认 hold

        # 4. 三路融合
        fused = fuser.fuse(
            rl_decision=rl_decision,
            rule_decision=rule_decision,
            llm_skills_decision=fusion_llm,
            market_state=market_label,
        )
        fused["code"] = code
        fused_list.append(fused)

        recommendations.append({
            "code": code,
            "last_close": analysis["last_close"],
            "indicators_snapshot": {
                "MA5": analysis["indicators"].get("MA5", 0),
                "RSI": analysis["indicators"].get("RSI", 50),
                "MACD_DIF": analysis["indicators"].get("MACD_DIF", 0),
                "volume_ratio": analysis["indicators"].get("VOL_RATIO", 1),
            },
            "skills_analysis": {
                "action": skills_decision["action"],
                "confidence": skills_decision["confidence"],
                "reasoning": skills_decision.get("analysis_text", "")[:200],
                "key_signals": skills_decision.get("key_signals", []),
            },
            "llm_analysis": {
                "action": llm_decision["action"] if llm_decision else "hold",
                "confidence": llm_decision["confidence"] if llm_decision else 0.0,
                "reasoning": llm_decision.get("analysis_text", "")[:200] if llm_decision else "",
                "key_signals": llm_decision.get("key_signals", []) if llm_decision else [],
                "disabled": llm_decision is None,
            } if llm_decision else None,
            "llm_skills": {  # 向后兼容：旧键名保留给融合用
                "action": fusion_llm["action"],
                "confidence": fusion_llm["confidence"],
                "reasoning": fusion_llm.get("analysis_text", fusion_llm.get("risk_note", ""))[:200],
                "key_signals": fusion_llm.get("key_signals", []),
            },
            "rules": {
                "signal": rule_decision["signal"],
                "strength": rule_decision["strength"],
                "top_signals": analysis["rule_engine"]["top_rules"],
            },
            "rl": rl_decision,
            "fused": {
                "action": fused["action"],
                "confidence": fused["confidence"],
                "position": fused["position"],
                "consensus": fused.get("consensus_level", "medium"),
                "is_consensus": fused.get("is_consensus", False),
            },
        })

    # 排名
    rankings = ranker.rank(fused_list) if fused_list else []

    return {
        "market_state": market_label,
        "suggested_position_pct": round(market.suggest_position_level() * 100),
        "recommendations": recommendations,
        "ranking": [
            {"rank": r["rank"], "code": r["code"], "action": r["action"],
             "score": r["fusion_score"], "confidence": r["confidence"]}
            for r in rankings
        ],
    }
