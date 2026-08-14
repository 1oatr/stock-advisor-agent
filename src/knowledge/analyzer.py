"""knowledge/analyzer.py — 规则分析器

对每只股票运行规则库，输出规则匹配结果和信号强度。
支持否决级规则（如止损），触发后直接覆盖综合信号。
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd

from .rules import RULE_REGISTRY, VETO_RULES, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


class RuleAnalyzer:
    """规则分析器

    对给定股票数据运行所有规则，计算综合信号。

    两阶段流程：
    1. 投票规则 (RULE_REGISTRY) → 加权投票 → 综合信号
    2. 否决规则 (VETO_RULES) → 触发则覆盖综合信号
    """

    def __init__(self):
        self.rule_results: Dict[str, list] = {}

    def analyze(self, df: pd.DataFrame, code: str = "",
                entry_price: float = 0.0, market_state: str = "range",
                exclude_rules: list = None) -> Dict:
        """对一只股票运行所有规则

        Args:
            df: 股票数据（含技术指标）
            code: 股票代码
            entry_price: 持仓成本价（用于止损规则，0=无持仓）
            market_state: 市场状态（bull/range/bear），用于动态调整规则权重
            exclude_rules: 要跳过的规则名列表（如港股/美股禁 limit_gap）

        Returns:
            {
                "code": "000001",
                "rules": [
                    {"name": "ma_trend", "signal": "buy", "strength": 0.8, "weight": 0.22, "explanation": "..."},
                    ...
                ],
                "composite_signal": "buy",
                "composite_strength": 0.72,
                "consensus": 0.6,
                "veto_triggered": false,
            }
        """
        # ── 阶段 1：运行投票规则 ──
        results = []
        # 根据市场状态动态调整权重
        adjusted_weights = self._adjust_weights_for_market(market_state)
        # 获取自适应阈值
        thresholds = self.ADAPTIVE_THRESHOLDS.get(market_state,
                                                   self.ADAPTIVE_THRESHOLDS["range"])

        for rule_def in RULE_REGISTRY:
            try:
                # 需要 code 的规则传入股票代码
                fn = rule_def["fn"]
                rule_name = rule_def["name"]
                if exclude_rules and rule_name in exclude_rules:
                    continue  # 跳过被排除的规则（如港股/美股涨跌停）

                if rule_name == "limit_gap":
                    signal, strength, explanation = fn(df, code=code)
                elif rule_name == "rsi":
                    signal, strength, explanation = fn(
                        df,
                        oversold=thresholds["rsi_oversold"],
                        overbought=thresholds["rsi_overbought"],
                    )
                else:
                    signal, strength, explanation = fn(df)
                effective_weight = adjusted_weights.get(rule_name, rule_def["weight"])
                results.append({
                    "name": rule_def["name"],
                    "signal": signal,
                    "strength": round(float(strength), 4),
                    "weight": round(effective_weight, 4),
                    "explanation": explanation,
                })
            except Exception as e:
                results.append({
                    "name": rule_def["name"],
                    "signal": SIGNAL_HOLD,
                    "strength": 0.0,
                    "weight": rule_def["weight"],
                    "explanation": f"规则执行异常: {e}",
                })

        composite_signal, composite_strength, consensus = self._calculate_composite(results)

        # ── 阶段 2：运行否决规则 ──
        veto_triggered = False
        veto_info = None
        for veto_def in VETO_RULES:
            try:
                if veto_def["name"] == "stop_loss":
                    signal, strength, explanation = veto_def["fn"](df, entry_price=entry_price)
                else:
                    signal, strength, explanation = veto_def["fn"](df)

                if signal != SIGNAL_HOLD and strength > 0:
                    veto_triggered = True
                    veto_info = {
                        "name": veto_def["name"],
                        "signal": signal,
                        "strength": strength,
                        "explanation": explanation,
                    }
                    # 否决规则覆盖综合信号
                    composite_signal = signal
                    composite_strength = strength
                    break  # 只触发第一个否决规则
            except Exception as e:
                pass  # 否决规则异常不影响主流程

        output = {
            "code": code,
            "rules": results,
            "composite_signal": composite_signal,
            "composite_strength": round(composite_strength, 4),
            "consensus": round(consensus, 4),
            "veto_triggered": veto_triggered,
        }
        if veto_info:
            output["veto_info"] = veto_info

        self.rule_results[code] = results
        return output

    def analyze_multi(self, data_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """批量分析多只股票

        Args:
            data_dict: {code: DataFrame}

        Returns:
            每只股票的分析结果列表
        """
        return [self.analyze(df, code) for code, df in data_dict.items()]

    def format_result(self, result: Dict) -> str:
        """格式化分析结果为可读文本"""
        lines = [f"\n📐 规则引擎分析 [{result.get('code', '')}]", "=" * 50]

        signal_map = {"buy": "🟢买入", "sell": "🔴卖出", "hold": "⚪持有"}
        composite = result.get("composite_signal", "hold")
        strength = result.get("composite_strength", 0)
        consensus = result.get("consensus", 0)

        # 否决标记
        veto_note = ""
        if result.get("veto_triggered") and result.get("veto_info"):
            vi = result["veto_info"]
            veto_note = f" ⚠️ 否决规则触发: [{vi['name']}] {vi.get('explanation', '')}"

        lines.append(f"综合信号: {signal_map.get(composite, '⚪持有')} "
                     f"(强度: {strength:.0%}, 共识度: {consensus:.0%}){veto_note}")
        lines.append("")

        for rule in result.get("rules", []):
            icon = {"buy": "✅", "sell": "❌", "hold": "➖"}
            lines.append(f"  {icon.get(rule['signal'], '➖')} "
                         f"{rule['name']}: {rule.get('explanation', '')}")

        lines.append("=" * 50)
        return "\n".join(lines)

    # ========================================================================
    # 动态权重调整
    # ========================================================================

    # 规则分类（用于相关性控制和市场自适应）
    RULE_CATEGORIES = {
        "trend": ["ma_trend", "macd", "adx"],              # 趋势跟踪类
        "oscillator": ["kdj", "rsi"],                       # 震荡指标类
        "channel": ["bollinger"],                           # 通道/波动类
        "volume": ["volume_price", "volume_anomaly"],       # 量价类
        "structure": ["support_resistance", "limit_gap"],   # 价格结构类
    }

    # 不同市场状态下各类规则的权重乘数
    # 趋势市 → 趋势规则更可信；震荡市 → 震荡指标和通道规则更可信
    MARKET_WEIGHT_MODIFIERS = {
        "bull": {
            "trend": 1.25,       # 牛市中趋势规则加权重
            "oscillator": 0.80,  # 牛市RSI/KDJ可能持续超买，降权
            "channel": 0.85,     # 牛市布林上轨可能持续被突破
            "volume": 1.10,      # 牛市量价配合更可靠
            "structure": 1.00,
        },
        "range": {
            "trend": 0.75,       # 震荡市趋势规则假信号多，降权
            "oscillator": 1.30,  # 震荡市超买超卖最有效
            "channel": 1.20,     # 震荡市布林带上下轨更可靠
            "volume": 0.90,
            "structure": 1.10,
        },
        "bear": {
            "trend": 1.10,       # 熊市趋势向下，趋势规则仍有效
            "oscillator": 0.85,  # 熊市RSI/KDJ可能持续超卖，降权
            "channel": 0.80,     # 熊市布林下轨可能持续被跌破
            "volume": 0.85,      # 熊市放量下跌更可靠但需谨慎
            "structure": 1.05,
        },
    }

    # 市场自适应阈值：不同市场状态下动态调整超买超卖阈值
    # 牛市：超买阈值上移（牛市不言顶），超卖阈值上移（回调即买点）
    # 熊市：超买阈值下移（反弹即卖点），超卖阈值下移（熊市不言底）
    ADAPTIVE_THRESHOLDS = {
        "bull": {
            "rsi_oversold": 35, "rsi_overbought": 75,     # 牛市中 RSI 70→75 超买，30→35 超卖
            "boll_oversold": 0.22, "boll_overbought": 0.78,  # %B 阈值放宽
        },
        "range": {
            "rsi_oversold": 30, "rsi_overbought": 70,      # 震荡市使用标准阈值
            "boll_oversold": 0.30, "boll_overbought": 0.70,
        },
        "bear": {
            "rsi_oversold": 25, "rsi_overbought": 65,      # 熊市中 RSI 70→65 超买，30→25 超卖
            "boll_oversold": 0.18, "boll_overbought": 0.62,  # %B 阈值收紧
        },
    }

    def _adjust_weights_for_market(self, market_state: str) -> Dict[str, float]:
        """根据市场状态动态调整规则权重

        趋势市增强趋势跟踪类规则，震荡市增强均值回归类规则。
        总权重可能 > 1.0，在 _calculate_composite 中会重新归一化。
        """
        modifiers = self.MARKET_WEIGHT_MODIFIERS.get(market_state,
                                                      self.MARKET_WEIGHT_MODIFIERS["range"])
        adjusted = {}
        for rule_def in RULE_REGISTRY:
            name = rule_def["name"]
            base_weight = rule_def["weight"]
            # 找到规则所属类别
            category = None
            for cat, rule_names in self.RULE_CATEGORIES.items():
                if name in rule_names:
                    category = cat
                    break
            modifier = modifiers.get(category, 1.0) if category else 1.0
            adjusted[name] = base_weight * modifier
        return adjusted

    # ========================================================================
    # 综合信号计算
    # ========================================================================

    def _calculate_composite(self, results: list) -> Tuple[str, float, float]:
        """计算综合信号、强度和共识度

        改进点（相比旧版简单加权）：
        1. 按类别分组后再加权 → 避免同类别相关规则重复计数
        2. 类别内取最强信号 → 趋势类的 MA 和 MACD 不重复加分
        3. 类别间加权投票 → 不同维度各说各的
        4. 共识度考虑所有非 hold 信号 + hold 信号的"弃权"影响
        """
        # 按类别分组
        categories: Dict[str, list] = {}
        for r in results:
            name = r.get("name", "")
            # 找到所属类别
            cat = None
            for c_name, rule_names in self.RULE_CATEGORIES.items():
                if name in rule_names:
                    cat = c_name
                    break
            if cat is None:
                cat = "other"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        # 类别内：取最强买入分和最强卖出分（不重复计数）
        # 同时收集所有非 hold 信号用于共识度计算
        category_scores = {}  # {cat: {"buy": max_buy, "sell": max_sell, "weight": total_weight}}
        all_signals = []

        for cat, cat_rules in categories.items():
            max_buy = 0.0
            max_sell = 0.0
            total_w = 0.0
            for r in cat_rules:
                w = r.get("weight", 1.0)
                s = r.get("strength", 0.0)
                sig = r.get("signal", SIGNAL_HOLD)
                total_w += w
                all_signals.append(sig)

                if sig == SIGNAL_BUY:
                    max_buy = max(max_buy, w * s)
                elif sig == SIGNAL_SELL:
                    max_sell = max(max_sell, w * s)
                # hold 不贡献分数，但已计入 all_signals

            category_scores[cat] = {
                "buy": max_buy,
                "sell": max_sell,
                "weight": total_w,
            }

        # 类别间：加权合计
        total_weight = sum(v["weight"] for v in category_scores.values())
        buy_score = sum(v["buy"] for v in category_scores.values())
        sell_score = sum(v["sell"] for v in category_scores.values())

        if total_weight == 0:
            return SIGNAL_HOLD, 0.5, 0.5

        if buy_score > sell_score:
            composite = SIGNAL_BUY
            strength = buy_score / total_weight
        elif sell_score > buy_score:
            composite = SIGNAL_SELL
            strength = sell_score / total_weight
        else:
            composite = SIGNAL_HOLD
            strength = 0.5

        # 共识度：考虑所有有效信号
        # 改进：hold 算作"弃权"，不算入总分母但降低共识度
        non_hold = [s for s in all_signals if s != SIGNAL_HOLD]
        n_total = len(all_signals)
        if non_hold:
            # 方向一致性 × 参与率
            direction_ratio = max(non_hold.count(SIGNAL_BUY), non_hold.count(SIGNAL_SELL)) / len(non_hold)
            participation = len(non_hold) / n_total  # 有多少规则给出了明确方向
            consensus = direction_ratio * (0.5 + 0.5 * participation)
        else:
            consensus = 0.5

        return composite, strength, consensus
