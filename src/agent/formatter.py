"""agent/formatter.py — 结果格式化

将工具执行结果格式化为用户可读的输出。
支持 rich 表格、三路决策分开展示、分歧标注。
"""

import json
from typing import Dict, List
from .memory import SessionMemory


class Formatter:
    """结果格式化器"""

    def __init__(self, memory: SessionMemory = None):
        self.memory = memory or SessionMemory()

    # ========================================================================
    # 辅助方法
    # ========================================================================

    @staticmethod
    def _confidence_label(conf: float) -> str:
        """将 0~1 置信度转为易懂的文字等级"""
        if conf >= 0.70:
            return "高把握"
        elif conf >= 0.50:
            return "中等把握"
        else:
            return "低把握"

    @staticmethod
    def _advice_strength(conf: float) -> str:
        """将置信度转为建议力度"""
        if conf >= 0.70:
            return "强烈建议"
        elif conf >= 0.50:
            return "建议"
        else:
            return "略微考虑"

    @staticmethod
    def _position_size(conf: float, pos_pct: float = None) -> str:
        """仓位对应的大小描述"""
        if pos_pct is not None:
            if pos_pct >= 0.15:
                return "大量"
            elif pos_pct >= 0.08:
                return "适量"
        # 无明确仓位时，根据把握度推断
        if conf >= 0.70:
            return "适量"
        return "少量"

    @staticmethod
    def _prob_label(prob: str) -> str:
        """将预测概率转为更口语化的表达"""
        mapping = {"大概率": "很可能", "中等概率": "较可能", "小概率": "不太可能"}
        return mapping.get(prob, prob)

    @staticmethod
    def _action_label(action: str) -> str:
        """将 buy/sell/hold 转为中文操作"""
        mapping = {"buy": "增持", "sell": "减持", "hold": "持有"}
        return mapping.get(action, action)

    @staticmethod
    def _full_advice(action: str, conf: float, pos_pct: float = None) -> str:
        """生成完整操作建议，如 '强烈建议适量增持' 或 '不建议增持'"""
        if action == "hold":
            if conf < 0.40:
                return "不建议操作，维持持有"
            return "建议维持持有，暂不操作"

        strength = "强烈建议" if conf >= 0.70 else ("建议" if conf >= 0.50 else "略微考虑")
        size = Formatter._position_size(conf, pos_pct)
        op = "增持" if action == "buy" else "减持"

        # 低把握时用否定形式：不建议XX
        if conf < 0.40:
            return f"不建议{op}，信号不明确"
        return f"{strength}{size}{op}"

    def format_analyze_result(self, result: dict) -> str:
        """格式化单股分析结果"""
        if "error" in result:
            return f"  ❌ {result.get('code', '?')}: {result['error']}"

        rule = result.get("rule_engine", {})
        skills = result.get("skills", {})
        ind = result.get("indicators", {})

        lines = [
            f"",
            f"═══ {result['code']} 技术分析 ═══",
            f"  收盘: {result['last_close']:.2f}  |  "
            f"{result.get('period_return', 0):+.1f}% ({result.get('data_days', '?')}天)",
            f"  52周: {result.get('low_52w', '?')} ~ {result.get('high_52w', '?')}",
            f"",
            f"  📈 关键指标",
            f"  MA5={ind.get('MA5','?')} MA20={ind.get('MA20','?')} MA60={ind.get('MA60','?')}",
            f"  RSI={ind.get('RSI','?')} | MACD: DIF={ind.get('MACD_DIF','?')} DEA={ind.get('MACD_DEA','?')}",
            f"  布林: 上{ind.get('BOLL_UP','?')} 中{ind.get('BOLL_MID','?')} 下{ind.get('BOLL_DN','?')} %B={ind.get('BOLL_POSITION','?')}",
            f"  量比={ind.get('VOL_RATIO','?')} | ATR%={ind.get('ATR_PCT','?')} | CCI={ind.get('CCI','?')}",
            f"",
            f"  📐 规则引擎: {self._full_advice(rule.get('composite_signal','hold'), rule.get('composite_strength',0))}",
        ]

        for r in rule.get("top_rules", [])[:3]:
            icon = {"buy": "✅", "sell": "❌", "hold": "➖"}
            lines.append(f"    {icon.get(r['signal'], '➖')} {r['name']}: {r.get('explanation', '')}")

        lines.append(f"")
        lines.append(f"  📊 技能汇总: {self._full_advice(skills.get('aggregate_signal','hold'), skills.get('confidence',0))}")
        if skills.get("buy_skills"):
            lines.append(f"    🟢 增持信号: {', '.join(skills['buy_skills'])}")
        if skills.get("sell_skills"):
            lines.append(f"    🔴 减持信号: {', '.join(skills['sell_skills'])}")

        lines.append(f"")
        return "\n".join(lines)

    def format_skills_analyze_result(self, result: dict) -> str:
        """格式化本地策略分析结果（非LLM，5维评分卡 + 统计预测）"""
        if "error" in result:
            return f"  ⚠️ 本地策略异常: {result.get('error', '')}"

        action = result.get("action", "hold")
        conf = result.get("confidence", 0.5)
        advice = self._full_advice(action, conf)

        lines = [
            f"  📊 本地策略分析: {advice}",
            f"      ── {result.get('analysis_text', '')[:200]}",
        ]

        key_signals = result.get("key_signals", [])
        if key_signals:
            lines.append(f"      关键信号: {', '.join(key_signals[:4])}")

        # ---- 价格预测 ----
        predictions = result.get("predictions", {})
        if predictions:
            lines.append(f"")
            lines.append(f"  📈 统计预测:")
            for span, key in [("短期", "short_term"), ("中期", "mid_term"),
                  ("中长期", "mid_long_term"), ("长期", "long_term")]:
                pred = predictions.get(key, {})
                if pred:
                    direction = pred.get("direction", "?")
                    prob_raw = pred.get("probability", "?")
                    prob = self._prob_label(prob_raw)
                    pct = pred.get("change_pct", 0)
                    days = pred.get("days", "?")
                    reason = pred.get("reason", "")
                    if direction == "震荡":
                        arrow = "➡️"
                        move_desc = "横盘整理"
                    else:
                        arrow = "📈" if pct > 0 else "📉"
                        move_desc = f"{direction}{abs(pct):.1f}%"
                    lines.append(
                        f"    {arrow} {span}({days}): {prob}{move_desc}，{reason}"
                    )

        risk = result.get("risk_note", "")
        if risk:
            lines.append(f"")
            lines.append(f"      ⚠️ 风险: {risk[:150]}")

        return "\n".join(lines)

    def format_llm_analyze_result(self, result: dict) -> str:
        """格式化 LLM 深度分析结果（含价格预测 + 新闻舆情）"""
        if result.get("disabled"):
            return ""  # LLM 关闭时占位结果不展示（skills_analyze 已覆盖）

        if "error" in result:
            return f"  ⚠️ LLM分析异常: {result.get('error', '')}"

        action = result.get("action", "hold")
        conf = result.get("confidence", 0.5)
        advice = self._full_advice(action, conf)

        lines = [
            f"  🧠 LLM 深度分析: {advice}",
            f"      ── {result.get('analysis_text', '')[:200]}",
        ]

        key_signals = result.get("key_signals", [])
        if key_signals:
            lines.append(f"      关键信号: {', '.join(key_signals[:4])}")

        # ---- 新闻舆情 ----
        news_sentiment = result.get("news_sentiment", "")
        if news_sentiment:
            lines.append(f"")
            lines.append(f"  📰 近期新闻舆情: {news_sentiment[:200]}")

        # ---- 价格预测 ----
        predictions = result.get("predictions", {})
        if predictions:
            lines.append(f"")
            lines.append(f"  📈 价格预测:")
            for span, key in [("短期", "short_term"), ("中期", "mid_term"),
                  ("中长期", "mid_long_term"), ("长期", "long_term")]:
                pred = predictions.get(key, {})
                if pred:
                    direction = pred.get("direction", "?")
                    prob_raw = pred.get("probability", "?")
                    prob = self._prob_label(prob_raw)
                    pct = pred.get("change_pct", 0)
                    days = pred.get("days", "?")
                    reason = pred.get("reason", "")
                    if direction == "震荡":
                        arrow = "➡️"
                        move_desc = "横盘整理"
                    else:
                        arrow = "📈" if pct > 0 else "📉"
                        move_desc = f"{direction}{abs(pct):.1f}%"
                    lines.append(
                        f"    {arrow} {span}({days}): {prob}{move_desc}，{reason}"
                    )

        risk = result.get("risk_note", "")
        if risk:
            lines.append(f"")
            lines.append(f"      ⚠️ 风险: {risk[:150]}")

        source = result.get("source", "")
        if source == "fallback_skills":
            lines.append(f"      [本地回退模式，未使用LLM]")

        return "\n".join(lines)

    def format_fused_result(self, recommendations: list, market_state: str = "unknown") -> str:
        """格式化三路融合预测结果 — 三路各说各的 + 合并结论"""
        if not recommendations:
            return "  (无预测结果)"

        state_icon = {"bull": "🚀", "bear": "📉", "range": "📊"}
        icon = state_icon.get(market_state, "❓")

        lines = [f"", f"═══ 三路融合预测 (大盘: {icon} {market_state}) ═══", ""]

        for rec in recommendations:
            if "error" in rec:
                lines.append(f"  ❌ {rec['code']}: {rec['error']}")
                continue

            code = rec["code"]
            skills = rec.get("skills_analysis", {})
            llm = rec.get("llm_analysis") or {}
            rules = rec.get("rules", {})
            rl = rec.get("rl", {})
            fused = rec.get("fused", {})

            a_icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}

            # 各路独立意见
            lines.append(f"  ┌─ {code} ─────────────────────────────┐")

            # 本地策略分析（始终展示）
            if skills:
                skills_conf = skills.get('confidence', 0)
                lines.append(f"  │ 📊 本地策略:  {a_icon.get(skills.get('action','hold'),'⚪')} {self._full_advice(skills.get('action','hold'), skills_conf)}")

            # LLM 深度分析（仅在 LLM ON 时展示）
            if llm and not llm.get("disabled"):
                llm_conf = llm.get('confidence', 0)
                lines.append(f"  │ 🧠 LLM 分析:  {a_icon.get(llm.get('action','hold'),'⚪')} {self._full_advice(llm.get('action','hold'), llm_conf)}")
            elif llm and llm.get("disabled"):
                lines.append(f"  │ 🧠 LLM 分析:  🔒 已关闭")

            # 规则
            lines.append(f"  │ 📐 硬规则:    {a_icon.get(rules.get('signal','hold'),'⚪')} {self._full_advice(rules.get('signal','hold'), rules.get('strength',0))}")

            # RL
            rl_untrained = rl.get("untrained", False)
            if rl_untrained:
                rl_display = "未训练，无法给出建议"
            else:
                rl_conf = rl.get('confidence', 0)
                rl_display = f"{a_icon.get(rl.get('action','hold'),'⚪')} {self._full_advice(rl.get('action','hold'), rl_conf)}"
            lines.append(f"  │ 🎮 RL智能体:  {rl_display}")

            # 融合结论
            consensus = fused.get("consensus", "medium")
            consensus_labels = {"high": "三路一致 ✅ 可信度高", "medium": "两路一致 ⚡ 可参考", "low": "三路分歧 ⚠️ 建议谨慎"}
            consensus_note = consensus_labels.get(consensus, "")
            pos = fused.get("position", 0)
            pos_note = ""
            if consensus == "low":
                pos_note = "，分歧较大建议减少操作"
            fused_conf = fused.get('confidence', 0)

            lines.append(f"  │")
            lines.append(f"  ├─ 🔗 最终建议: {a_icon.get(fused.get('action','hold'),'⚪')} {self._full_advice(fused.get('action','hold'), fused_conf, pos)}{pos_note}")
            lines.append(f"  │      三路共识: {consensus_note}")
            lines.append(f"  └──────────────────────────────────────┘")
            lines.append("")

        # 排名
        lines.append(f"  📊 推荐排序（按综合把握度从高到低）:")
        for i, rec in enumerate([r for r in recommendations if "error" not in r], 1):
            f = rec["fused"]
            lines.append(f"    {i}. {rec['code']} → {self._full_advice(f.get('action','hold'), f.get('confidence',0))}")

        lines.append("")
        return "\n".join(lines)

    def format_rl_result(self, result: dict) -> str:
        """格式化 RL 预测结果"""
        if result.get("untrained"):
            reason = result.get("reason", "模型未训练")
            return f"  🎮 RL智能体: 未训练（{reason}），无法给出建议\n  💡 试试 stock-advisor train {result.get('code', '?')}"

        action = result.get("action", "hold")
        conf = result.get("confidence", 0.5)
        advice = self._full_advice(action, conf)
        fresh = "✅" if result.get("model_fresh") else "⚠️ 模型较旧"
        details = result.get("details", {})

        lines = [
            f"  🎮 RL智能体: {advice} (新鲜度:{fresh})",
        ]
        if details:
            b = details.get("buy_ratio", 0)
            s = details.get("sell_ratio", 0)
            h = details.get("hold_ratio", 0)
            lines.append(f"      步内偏好: 买{b:.0%} 卖{s:.0%} 持{h:.0%}")
        return "\n".join(lines)

    def format_auto_fused_result(self, code: str, rule_decision: dict,
                                  llm_decision: dict, skills_decision: dict,
                                  rl_decision: dict, market_state: str = "range") -> str:
        """自动融合三路决策并格式化输出（用于单股分析后的自动汇总）

        展示所有独立分析结果（本地策略 + LLM + 规则 + RL），再给出融合建议。
        """
        try:
            from src.advisor.fuser import DecisionFuser
            fuser = DecisionFuser()
            rl_untrained = rl_decision.get("untrained", False)
            # 融合用 LLM 结果（如果有），否则用本地策略结果
            fusion_input = llm_decision if llm_decision else skills_decision
            if fusion_input is None:
                return ""
            fused = fuser.fuse(
                rl_decision=rl_decision,
                rule_decision=rule_decision,
                llm_skills_decision=fusion_input,
                market_state=market_state,
                rl_untrained=rl_untrained,
            )
        except Exception:
            return ""

        a_icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}
        consensus = fused.get("consensus_level", "medium")
        consensus_labels = {
            "high": "三路一致 ✅ 可信度高",
            "medium": "两路一致 ⚡ 可参考",
            "low": "三路分歧 ⚠️ 建议谨慎",
        }
        pos = fused.get("position", 0)
        pos_note = "，分歧较大建议减少操作" if consensus == "low" else ""
        fused_conf = fused.get("confidence", 0)

        lines = [
            f"",
            f"  ┌─ 🔗 三方融合结果 ─────────────────────┐",
        ]

        # 本地策略分析（始终展示）
        if skills_decision:
            lines.append(f"  │ 📊 本地策略:  {a_icon.get(skills_decision.get('action','hold'),'⚪')} {self._full_advice(skills_decision.get('action','hold'), skills_decision.get('confidence',0))}")

        # LLM 深度分析（仅在 LLM ON 时展示）
        if llm_decision:
            lines.append(f"  │ 🧠 LLM 分析:  {a_icon.get(llm_decision.get('action','hold'),'⚪')} {self._full_advice(llm_decision.get('action','hold'), llm_decision.get('confidence',0))}")

        # 规则
        lines.append(f"  │ 📐 硬规则:    {a_icon.get(rule_decision.get('signal','hold'),'⚪')} {self._full_advice(rule_decision.get('signal','hold'), rule_decision.get('strength',0))}")

        # RL
        rl_untrained = rl_decision.get("untrained", False)
        if rl_untrained:
            lines.append(f"  │ 🎮 RL智能体:  未训练，跳过此路")
        else:
            lines.append(f"  │ 🎮 RL智能体:  {a_icon.get(rl_decision.get('action','hold'),'⚪')} {self._full_advice(rl_decision.get('action','hold'), rl_decision.get('confidence',0))}")

        lines.append(f"  │")
        lines.append(f"  ├─ 🔗 最终建议: {a_icon.get(fused.get('action','hold'),'⚪')} {self._full_advice(fused.get('action','hold'), fused_conf, pos)}{pos_note}")
        lines.append(f"  │      三路共识: {consensus_labels.get(consensus, '?')}")
        lines.append(f"  └──────────────────────────────────────────┘")
        lines.append("")
        return "\n".join(lines)

    def format_cached_fusion_pending(self, code: str, rule_decision: dict,
                                      llm_decision: dict, skills_decision: dict) -> str:
        """RL 未训练时的缓存提示面板 — 标记各分析路的状态"""
        a_icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}

        lines = [
            f"",
            f"  ┌─ 🔗 三方融合 ──────────────────────────┐",
        ]

        # 本地策略 — 已计算
        if skills_decision:
            lines.append(f"  │ 📊 本地策略:  [已计算 ✅]                 │")
        else:
            lines.append(f"  │ 📊 本地策略:  [未执行]                    │")

        # LLM — 已计算或已关闭
        if llm_decision:
            lines.append(f"  │ 🧠 LLM 分析:  [已计算 ✅]                 │")
        else:
            lines.append(f"  │ 🧠 LLM 分析:  [未执行 / 已关闭]           │")

        # 规则 — 已计算
        if rule_decision:
            lines.append(f"  │ 📐 硬规则:    [已计算 ✅]                 │")
        else:
            lines.append(f"  │ 📐 硬规则:    [未执行]                    │")

        # RL — 待训练
        lines.append(f"  │ 🎮 RL智能体:  ⏳ 待训练                    │")
        lines.append(f"  │                                          │")
        lines.append(f"  │ 💡 分析结果已缓存，RL 训练完成后将自动展示 │")
        lines.append(f"  │    完整的三路融合结果                     │")
        lines.append(f"  └──────────────────────────────────────────┘")
        lines.append("")
        return "\n".join(lines)

    def format_scan_result(self, result: dict) -> str:
        """格式化扫描结果"""
        if "error" in result:
            return f"  ❌ 扫描失败: {result['error']}"

        state_icon = {"bull": "🚀", "bear": "📉", "range": "📊"}
        icon = state_icon.get(result.get("market_state", ""), "❓")

        lines = [
            f"",
            f"═══ 📡 市场扫描 ═══",
            f"  大盘: {icon} {result.get('market_state','?')} (20日: {result.get('index_return_20d',0):+.1f}%)",
            f"  建议仓位: {result.get('suggested_position_pct','?')}%",
            f"",
        ]

        stocks = result.get("hot_stocks", [])
        if not stocks:
            lines.append("  (无热门股数据)")
        else:
            lines.append(f"  热门 TOP {len(stocks)}:")
            for s in stocks:
                lines.append(f"    #{s['rank']:<3} {s['code']} {s.get('name',''):8s}  "
                             f"收盘{s['close']:>8.2f}  "
                             f"涨幅{s['momentum']:>+5.1f}%  量比{s.get('volume_ratio',0):.1f}")

        lines.append("")
        return "\n".join(lines)

    def format_compare_result(self, result: dict) -> str:
        """格式化对比结果"""
        if "error" in result:
            return f"  ❌ 对比失败: {result['error']}"

        lines = [f"", f"═══ 多股对比 ═══"]
        table = result.get("comparison_table", [])
        if not table:
            return "\n".join(lines + ["  (无数据)"])

        for r in table:
            if "error" in r:
                lines.append(f"  ❌ {r['code']}: {r['error']}")
                continue
            signal_icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}
            lines.append(
                f"  {signal_icon.get(r['rule_signal'],'⚪')} {r['code']:6s} "
                f"收盘{r['close']:>8.2f}  "
                f"涨跌{r['period_return_pct']:>+6.1f}%  "
                f"RSI={r['RSI']:>5.1f}  "
                f"规则={r['rule_signal']}({r['rule_strength']:.0%})  "
                f"技能={r['skill_signal']}({r['skill_confidence']:.0%})"
            )

        verdict = result.get("verdict", {})
        if verdict.get("best"):
            lines.append(f"\n  🏆 最强: {verdict['best']}")
        if verdict.get("worst"):
            lines.append(f"  ⚠️  最弱: {verdict['worst']}")

        lines.append(f"  📝 {result.get('summary', '')}")
        lines.append("")
        return "\n".join(lines)

    def format_train_result(self, result: dict) -> str:
        """格式化训练结果"""
        if result.get("status") == "error":
            return f"  ❌ 训练失败: {result.get('message', '')}"

        if result.get("status") == "already_fresh":
            return f"  ✅ {result['code']} 模型已是最新（15天内训练过），无需更新"

        mode = "增量更新" if result.get("mode") == "incremental" else "全量训练"
        metrics = result.get("eval_metrics", {})
        early = " ⏩ 提前停止" if result.get("early_stopped") else ""
        train_rows = result.get("train_rows", result.get("data_rows", "?"))
        val_rows = result.get("val_rows", 0)

        lines = [
            f"",
            f"═══ 🎮 RL 训练完成 ═══",
            f"  股票: {result.get('code','?')} | 模式: {mode}{early}",
            f"  训练步数: {result.get('timesteps','?'):,} | 训练集: {train_rows}行 | 验证集: {val_rows}行",
            f"  训练集: {result.get('train_range', result.get('data_range','?'))}",
            f"  验证集: {result.get('val_range', '?')}",
        ]

        # 数据源状态
        data_sources = result.get("data_sources", {})
        if data_sources:
            src_parts = []
            for k, v in data_sources.items():
                label = {"daily_bars": "日线", "index": "大盘", "fund_flow": "资金"}.get(k, k)
                src_parts.append(f"{label}={v}")
            lines.append(f"  数据源: {', '.join(src_parts)}")

        # 警告
        warnings = result.get("warnings", [])
        if warnings:
            lines.append(f"")
            for w in warnings:
                lines.append(f"  ⚠️  {w}")

        if metrics:
            lines.append(f"")
            lines.append(f"  评估指标 (验证集):")
            lines.append(f"    总收益: {metrics.get('total_return_pct',0):+.1f}%")
            lines.append(f"    最大回撤: {metrics.get('max_drawdown_pct',0):.1f}%")
            lines.append(f"    夏普比: {metrics.get('sharpe',0)}")
            lines.append(f"    交易数: {metrics.get('trades',0)}")
            lines.append(f"    胜率:   {metrics.get('win_rate',0)}%")

        lines.append(f"  模型: {result.get('model_path','?')}")
        lines.append("")
        return "\n".join(lines)

    def format_backtest_result(self, result: dict) -> str:
        """格式化回测结果"""
        if "error" in result:
            return f"  ❌ 回测失败: {result['error']}"

        m = result.get("metrics", {})
        lines = [
            f"",
            f"═══ 📊 回测结果 ═══",
            f"  策略: {result['strategy']} | 标的: {result['stock_count']}只 | {result['date_range']}",
            f"  总收益: {m.get('total_return_pct',0):+.1f}%",
            f"  年化:   {m.get('annual_return_pct',0):+.1f}%",
            f"  最大回撤: {m.get('max_drawdown_pct',0):.1f}%",
            f"  夏普比: {m.get('sharpe_ratio',0)}",
            f"  胜率:   {m.get('win_rate_pct',0):.0f}%",
            f"  交易:   {m.get('total_trades',0)}笔",
            f"  最终资产: {m.get('final_equity',0):,.0f}",
            f"",
        ]
        return "\n".join(lines)

    def choose_formatter(self, tool_name: str, result: dict) -> str:
        """根据工具名选择格式化器"""
        formatters = {
            "analyze_stock": self.format_analyze_result,
            "skills_analyze": self.format_skills_analyze_result,
            "llm_analyze": self.format_llm_analyze_result,
            "predict_stocks": lambda r: self.format_fused_result(
                r.get("recommendations", []),
                r.get("market_state", "unknown"),
            ),
            "scan_market": self.format_scan_result,
            "compare_stocks": self.format_compare_result,
            "train_model": self.format_train_result,
            "run_backtest": self.format_backtest_result,
            "get_rl_prediction": self.format_rl_result,
            "get_market_state": lambda r: (
                f"  大盘: {r.get('market_state','?')} "
                f"(20日: {r.get('ret_20d',0):+.1f}%) "
                f"建议仓位: {r.get('suggested_position','?')}%"
            ),
        }

        fmt = formatters.get(tool_name)
        if fmt:
            return fmt(result)
        return json.dumps(result, ensure_ascii=False, indent=2)
