"""tools/compare_tool.py — 多股对比工具"""

from .analyze_tool import analyze_stock


def compare_stocks(codes: list) -> dict:
    """对比多只股票的技术面

    Returns:
        {
            "comparison_table": [
                {"code": "600519", "close": ..., "return": ..., "rsi": ..., ...},
                ...
            ],
            "summary": "基于...",
            "verdict": {"best_buy": "600519", "worst": "000858"}
        }
    """
    results = []
    for code in codes:
        analysis = analyze_stock(code)
        if "error" in analysis:
            results.append({"code": code, "error": analysis["error"]})
            continue

        indicators = analysis["indicators"]
        results.append({
            "code": code,
            "close": analysis["last_close"],
            "period_return_pct": analysis["period_return"],
            "MA5": indicators.get("MA5", 0),
            "MA20": indicators.get("MA20", 0),
            "RSI": indicators.get("RSI", 50),
            "MACD_DIF": indicators.get("MACD_DIF", 0),
            "volume_ratio": indicators.get("VOL_RATIO", 1),
            "rule_signal": analysis["rule_engine"]["composite_signal"],
            "rule_strength": analysis["rule_engine"]["composite_strength"],
            "skill_signal": analysis["skills"]["aggregate_signal"],
            "skill_confidence": analysis["skills"]["confidence"],
        })

    if not results:
        return {"error": "无有效数据", "codes": codes}

    # 根据规则强度排序（buy > sell > hold）
    def sort_key(r):
        if "error" in r:
            return -999
        s = r["rule_strength"]
        if r["rule_signal"] == "buy":
            return s
        elif r["rule_signal"] == "sell":
            return -s
        return 0

    results.sort(key=sort_key, reverse=True)

    # 最佳/最差
    best = results[0] if results else None
    worst = results[-1] if len(results) > 1 else None

    # 生成对比摘要
    summary_parts = []
    if best and "error" not in best:
        summary_parts.append(f"{best['code']} 技术面最强（规则={best['rule_signal']} RSI={best['RSI']}）")
    if worst and "error" not in worst and worst != best:
        summary_parts.append(f"{worst['code']} 最弱（规则={worst['rule_signal']}）")

    return {
        "comparison_table": results,
        "summary": "；".join(summary_parts) if summary_parts else "数据不足以判断",
        "verdict": {
            "best": best["code"] if best else None,
            "worst": worst["code"] if worst else None,
        },
        "stock_count": len([r for r in results if "error" not in r]),
    }
