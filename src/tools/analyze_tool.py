"""tools/analyze_tool.py — 单股深度分析工具

封装：数据获取 → 指标计算 → 规则引擎 → 技能分析 → 结构化结果
"""

import pandas as pd
from src.data.fetcher import DataFetcher
from src.data.indicators import add_all_indicators
from src.data.cleaning import DataCleaner
from src.knowledge.analyzer import RuleAnalyzer
from src.skills import init_skills, get_manager


def analyze_stock(code: str, days: int = 500, market: str = "a",
                  df: pd.DataFrame = None) -> dict:
    """深度技术分析单只股票

    Args:
        code: 股票代码
        days: 回看天数
        market: 市场 'a'/'hk'/'us'
        df: 预取的日线 DataFrame（多市场时由 WebUI 注入）

    Returns:
        {
            "code": "600519",
            "last_close": 1292.01,
            "period_return": -1.30,
            "indicators": {"MA5": ..., "RSI": ..., "MACD_DIF": ..., ...},
            "rule_engine": {"composite_signal": "sell", "strength": 0.15, ...},
            "skills": {"aggregate_signal": "sell", "confidence": 0.25, ...},
        }
    """
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    analyzer = RuleAnalyzer()
    init_skills()
    skill_mgr = get_manager()

    # 获取数据（外部注入 df 时跳过内部 fetch）
    if df is None:
        df = fetcher.daily_bars(code, start="", end="")
    if df.empty:
        return {"error": f"未获取到 {code} 数据", "code": code}

    df = cleaner.clean_single(df, code)
    df = add_all_indicators(df)

    if df.empty or len(df) < 20:
        return {"error": f"{code} 数据不足", "code": code}

    # 非 A 股禁用涨跌停规则（港股/美股无涨跌停制度）
    exclude_rules = ["limit_gap"] if market != "a" else None

    recent_df = df.tail(min(days, len(df)))

    # 基础信息
    last = recent_df.iloc[-1]
    first = recent_df.iloc[0]
    period_return = round((last["close"] / first["close"] - 1) * 100, 2)
    high_52w = float(df["high"].tail(252).max()) if len(df) > 20 else float(last["high"])
    low_52w = float(df["low"].tail(252).min()) if len(df) > 20 else float(last["low"])

    # 关键指标快照
    indicators = {}
    for col, label in [
        ("MA5", "MA5"), ("MA20", "MA20"), ("MA60", "MA60"),
        ("MACD_DIF", "MACD_DIF"), ("MACD_DEA", "MACD_DEA"), ("MACD_HIST", "MACD_HIST"),
        ("RSI", "RSI"), ("BOLL_UP", "BOLL_UP"), ("BOLL_MID", "BOLL_MID"), ("BOLL_DN", "BOLL_DN"),
        ("BOLL_WIDTH", "BOLL_WIDTH"), ("BOLL_POSITION", "BOLL_POSITION"),
        ("ATR_PCT", "ATR_PCT"), ("CCI", "CCI"), ("KDJ_K", "KDJ_K"), ("KDJ_D", "KDJ_D"),
        ("VOL_RATIO", "量比"),
    ]:
        candidates = [col, col.replace("RSI", "RSI14")]
        for c in candidates:
            if c in recent_df.columns:
                indicators[col] = round(float(recent_df[c].iloc[-1]), 2)
                break

    # 规则引擎
    rule_result = analyzer.analyze(recent_df, code, exclude_rules=exclude_rules)

    # 记录查询（长期记忆）
    from src.memory import record_stock_lookup
    record_stock_lookup(code)

    # 技能分析
    skill_results = skill_mgr.run_all(recent_df, code)
    skill_agg = skill_mgr.aggregate_signal(skill_results)

    # 简化技能详情
    skills_detail = []
    for sr in skill_results:
        skills_detail.append({
            "name": sr.skill_name,
            "signal": sr.signal,
            "confidence": round(sr.confidence, 2),
            "explanation": sr.explanation[:100],
        })

    return {
        "code": code,
        "last_close": round(float(last["close"]), 2),
        "last_volume": int(last["volume"]),
        "period_return": period_return,
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "data_days": len(recent_df),
        "data_range": f"{recent_df['date'].iloc[0].date()} ~ {recent_df['date'].iloc[-1].date()}",
        "indicators": indicators,
        "rule_engine": {
            "composite_signal": rule_result["composite_signal"],
            "composite_strength": rule_result["composite_strength"],
            "consensus": rule_result["consensus"],
            "top_rules": [
                {"name": r["name"], "signal": r["signal"], "strength": r["strength"],
                 "explanation": r["explanation"][:80]}
                for r in rule_result["rules"][:4]
            ],
        },
        "skills": {
            "aggregate_signal": skill_agg["signal"],
            "confidence": round(skill_agg["confidence"], 2),
            "buy_skills": skill_agg.get("buy_skills", []),
            "sell_skills": skill_agg.get("sell_skills", []),
            "details": skills_detail,
        },
    }
