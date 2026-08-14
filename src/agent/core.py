"""agent/core.py — Agent 核心主循环

初始化各组件，运行交互式 REPL。
用户输入自然语言 → Planner 调度工具 → Executor 执行 → Formatter 格式化输出。
"""

import sys
import os
import traceback
from pathlib import Path
from typing import Dict, Optional

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .planner import Planner
from .executor import Executor
from .formatter import Formatter
from .memory import SessionMemory
from src.tools.registry import register_all_tools, ToolRegistry


class AgentCore:
    """自主 CLI 智能体核心

    用法:
        agent = AgentCore()
        agent.run()          # 启动交互式 REPL
        agent.query("分析茅台")  # 单次查询（非交互）
    """

    WELCOME = r"""
╔══════════════════════════════════════════════════════╗
║        📊 Stock Advisor — 量化分析智能体              ║
║                                                      ║
║  三方决策: 🧠 LLM+Skills + 📐 规则 + 🎮 RL  ║
║  引擎: DeepSeek · 工具调度 · 会话记忆                  ║
║                                                      ║
║  输入自然语言开始分析  /help 查看帮助  /exit 退出       ║
╚══════════════════════════════════════════════════════════╝
"""

    HELP_TEXT = """
  ┌──── 软件介绍 ────────────────────────────────────┐
  │                                                  │
  │  Stock Advisor 是 A 股量化分析智能体，输入自然语言   │
  │  即可获得专业的股票买卖建议。                       │
  │                                                  │
  │  核心思路：三路决策融合                             │
  │    🧠 LLM+Skills — DeepSeek 深度解读11个技能         │
  │        结果 + 技术指标，理解信号间的全局关联          │
  │    📐 硬规则  — 均线/MACD/RSI/布林等6条规则         │
  │    🎮 RL智能体 — PPO单股模型，需先训练               │
  │                                                  │
  │  典型流程：                                       │
  │    1. 训练目标股票的 RL 模型（首次使用必须）          │
  │    2. 分析 → 看 LLM+Skills 的预测和建议             │
  │    3. 预测 → 三路融合给出最终买卖决策                │
  │    4. 觉得不够可对比其他股票，或回测验证策略          │
  │                                                  │
  ├──── 功能模块 ──────────────────────────────────┤
  │                                                  │
  │  分析单股    "分析茅台" / "茅台最近走势"             │
  │            → 技术指标 + 规则引擎 + 技能扫描 +       │
  │               LLM 深度解读 + 短期/长期价格预测       │
  │                                                  │
  │  多股对比    "对比茅台和五粮液" / "茅台vs招行哪个好"  │
  │            → 两股分头分析 + 多维度对比排名           │
  │                                                  │
  │  三路预测    "预测茅台和五粮液"                      │
  │            → LLM+规则+RL 三方融合，输出最终买卖建议   │
  │               (需先训练RL模型，否则RL路回退为hold)    │
  │                                                  │
  │  训练RL模型  "训练茅台" / "训练600519 20万步"       │
  │            → 用历史数据训练PPO智能体                │
  │               <15天增量更新，>60天自动过期重建        │
  │                                                  │
  │  扫描热门    "扫描热门" / "今天有什么热点"            │
  │            → 全市场最活跃股票TOP榜 + 大盘状态        │
  │                                                  │
  │  策略回测    "回测均线趋势策略"                      │
  │            → 含T+1/涨跌停/手续费/滑点的模拟交易       │
  │                                                  │
  │  大盘状态    "大盘怎么样"                            │
  │            → 指数涨跌 + 牛熊判断 + 建议仓位          │
  │                                                  │
  ├──── 内置命令 ──────────────────────────────────┤
  │                                                  │
  │    /help       📖 显示此帮助                       │
  │    /exit/quit  👋 退出程序                         │
  │    /clear      🧹 清空会话记忆                     │
  │    /history    📜 查看本轮分析历史                   │
  │    /tools      🔧 列出全部11个可用工具               │
  │    /llm on|off 🔛 开启/关闭 LLM（关闭时用本地增强引擎） │
  │    /interested ⭐ 查看感兴趣的股票（长期记忆）         │
  │    /debug      🐛 调试模式开关                      │
  └──────────────────────────────────────────────────┘
"""

    def __init__(
        self,
        api_key: str = None,
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
    ):
        self.memory = SessionMemory()
        self._running = False
        self._result_cache: Dict[str, list] = {}  # code → [(tool_name, result), ...]

        # 确定 API Key 并注入环境变量（确保所有工具都能读取）
        # ⚠️ 必须在 Executor/Planner 之前执行，否则 _has_api 未定义导致 crash
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if self.api_key:
            os.environ["DEEPSEEK_API_KEY"] = self.api_key
        self._has_api = bool(self.api_key)

        # 读取配置文件中的 LLM 默认开关
        config_use_llm = self._load_config_llm()

        # LLM 开关状态（优先级：无 key 强制关 > 全局配置 data/llm_config.json
        #   显式保存过 enabled > config.yaml agent.use_llm > 默认开启）
        from src.llm_config import llm_enabled_explicit
        explicit_enabled = llm_enabled_explicit()
        if not self._has_api:
            self.llm_enabled = False
        elif explicit_enabled is not None:
            self.llm_enabled = bool(explicit_enabled)  # WebUI 全局配置优先
        elif config_use_llm is not None:
            self.llm_enabled = config_use_llm  # 配置文件覆盖
        else:
            self.llm_enabled = True  # 有 API Key 且无配置时默认开启

        self.planner = Planner(api_key=api_key, api_base=api_base, model=model,
                               llm_enabled=self.llm_enabled)
        self.executor = Executor(memory=self.memory, llm_enabled=self.llm_enabled)
        self.formatter = Formatter(memory=self.memory)

        if not self._has_api:
            print("  ⚠️  未设置 DEEPSEEK_API_KEY，LLM 已自动关闭，使用本地增强策略")
            print("      设置方法: set DEEPSEEK_API_KEY=your_key  (Windows)")
            print("                export DEEPSEEK_API_KEY=your_key (Linux/Mac)")
            print("      设置后可输入 /llm on 开启 LLM 模式")
        elif not self.llm_enabled:
            print("  🔒 配置文件 config.yaml 中 agent.use_llm=false，LLM 默认关闭")
            print("      输入 /llm on 可临时开启")

        # 注册所有工具
        self._init_tools()

    def _init_tools(self):
        """初始化工具注册表"""
        try:
            register_all_tools()
        except Exception as e:
            print(f"  ⚠️  工具注册异常: {e}")

    def _load_config_llm(self) -> Optional[bool]:
        """从 config.yaml 读取 agent.use_llm 配置

        Returns:
            True/False 若配置存在，None 若配置文件不存在或字段缺失
        """
        try:
            import yaml
            config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            if not config_path.exists():
                return None
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config.get("agent", {}).get("use_llm")
        except Exception:
            return None  # 配置解析失败不影响启动

    # ========================================================================
    # REPL 主循环
    # ========================================================================

    def run(self):
        """启动交互式 REPL"""
        self._running = True
        print(self.WELCOME)

        while self._running:
            try:
                user_input = self._read_input()
                if user_input is None:
                    break  # EOF / Ctrl+D

                user_input = user_input.strip()
                if not user_input:
                    continue

                # 处理特殊命令
                if self._handle_command(user_input):
                    continue

                # 正常查询
                self.query(user_input)

            except KeyboardInterrupt:
                print("\n\n  再见!")
                break
            except Exception as e:
                print(f"  [ERROR] 系统异常: {e}")
                if os.getenv("STOCK_ADVISOR_DEBUG"):
                    traceback.print_exc()

    def query(self, user_input: str) -> str:
        """单次查询（非交互模式也可用）

        Args:
            user_input: 用户自然语言输入

        Returns:
            格式化的结果字符串
        """
        # 记录到记忆
        self.memory.add_user_message(user_input)

        # 获取工具 schema + 上下文
        tools_schema = ToolRegistry.to_openai_schema()
        context = self.memory.get_context_for_llm()

        # Phase 1: Planner — 决定调用哪些工具
        print(f"\n  🤔 分析中...")
        tool_calls = self.planner.plan(user_input, tools_schema, context)

        if not tool_calls:
            # LLM 没有选择工具，可能是纯对话
            msg = "  没有需要执行的操作。试试：分析茅台、扫描热门、对比茅台和五粮液"
            print(msg)
            self.memory.add_assistant_message(msg)
            return msg

        # 安全兜底：对比类问题自动补齐 compare_stocks
        compare_keywords = ["对比", "比较", "vs", "哪个好"]
        is_compare_query = any(w in user_input for w in compare_keywords)
        if is_compare_query:
            called_names = [c["name"] for c in tool_calls]
            if "compare_stocks" not in called_names:
                # 从已有的 analyze_stock 调用中提取 codes
                analyzed_codes = list(dict.fromkeys(
                    c["params"]["code"] for c in tool_calls
                    if c["name"] == "analyze_stock" and "code" in c["params"]
                ))
                if len(analyzed_codes) >= 2:
                    tool_calls.append({"name": "compare_stocks", "params": {"codes": analyzed_codes}})

        # 安全兜底：单股分析自动补齐 RL 预测
        called_names = [c["name"] for c in tool_calls]
        has_analyze = "analyze_stock" in called_names
        has_skills = "skills_analyze" in called_names
        has_llm = "llm_analyze" in called_names
        has_rl = "get_rl_prediction" in called_names
        has_predict = "predict_stocks" in called_names
        if has_analyze and (has_skills or has_llm) and not has_rl and not has_predict:
            # 提取已调用的 stock codes
            codes_to_rl = list(dict.fromkeys(
                c["params"]["code"] for c in tool_calls
                if c["name"] in ("analyze_stock", "skills_analyze", "llm_analyze") and "code" in c["params"]
            ))
            for code in codes_to_rl:
                tool_calls.append({"name": "get_rl_prediction", "params": {"code": code}})

        # Phase 2: Executor — 执行工具
        print(f"  📋 将执行 {len(tool_calls)} 个步骤")
        results = self.executor.execute(tool_calls)

        # Phase 3: Formatter — 格式化输出
        output_parts = []
        for (name, result) in results:
            formatted = self.formatter.choose_formatter(name, result)
            output_parts.append(formatted)

        # Phase 4 (自动): 如果同时有规则 + LLM + RL 结果 → 自动融合展示
        fusion_output, untrained_codes = self._auto_fuse(results)
        if fusion_output:
            output_parts.append(fusion_output)

        final_output = "\n".join(output_parts)
        print(final_output)

        # 记录到记忆
        self.memory.add_assistant_message(final_output)

        # Phase 5 (交互): RL 未训练 → 询问用户是否训练
        if untrained_codes and self._running:
            self._prompt_train_rl(untrained_codes)

        return final_output

    def _auto_fuse(self, results: list, skip_cache: bool = False) -> tuple:
        """检测结果中是否有三路决策，有则自动融合

        Args:
            results: Executor 返回的 [(tool_name, result_dict), ...]
            skip_cache: True 时不缓存结果（用于训练后/拒绝训练后重新融合）

        Returns:
            (formatted_text, untrained_codes)
            untrained_codes: RL 未训练的股票代码列表（供后续询问训练）
        """
        analyze_result = None
        skills_result = None
        llm_result = None
        rl_result = None
        code = None

        for name, result in results:
            if name == "analyze_stock" and "error" not in result:
                analyze_result = result
                code = result.get("code")
            elif name == "llm_analyze" and "error" not in result and not result.get("disabled"):
                llm_result = result  # LLM 真正执行了
            elif name == "skills_analyze" and "error" not in result:
                skills_result = result
            elif name == "get_rl_prediction":
                rl_result = result

        # LLM 关闭时回退到 skills_analyze 结果
        if llm_result is None and skills_result is not None:
            llm_result = skills_result

        # 至少需要 rules + 至少一个分析结果
        if analyze_result is None or (llm_result is None and skills_result is None):
            return "", []

        # 提取规则决策
        rule_engine = analyze_result.get("rule_engine", {})
        rule_decision = {
            "signal": rule_engine.get("composite_signal", "hold"),
            "strength": rule_engine.get("composite_strength", 0.3),
        }

        # LLM 决策（用于融合权重最高的那路）
        llm_decision = {
            "action": llm_result.get("action", "hold"),
            "confidence": llm_result.get("confidence", 0.5),
        } if llm_result else None

        # 本地策略决策（独立展示）
        skills_decision = {
            "action": skills_result.get("action", "hold"),
            "confidence": skills_result.get("confidence", 0.5),
        } if skills_result else None

        # RL 决策
        if rl_result is None:
            rl_result = {"action": "hold", "confidence": 0.5, "untrained": True}
        rl_decision = {
            "action": rl_result.get("action", "hold"),
            "confidence": rl_result.get("confidence", 0.5),
            "untrained": rl_result.get("untrained", False),
        }

        output = self.formatter.format_auto_fused_result(
            code or "?", rule_decision, llm_decision, skills_decision, rl_decision
        )

        # 收集 RL 未训练的代码
        untrained = []
        for name, result in results:
            if name == "get_rl_prediction" and result.get("untrained"):
                # 从 params 或 result 中提取代码
                stock_code = result.get("code", "")
                if not stock_code:
                    # 从配对结果中推断
                    stock_code = code or ""
                if stock_code and stock_code not in untrained:
                    untrained.append(stock_code)

        # RL 未训练时：缓存全部结果，暂不显示融合（训练完后再展示完整融合）
        if untrained:
            if not skip_cache:
                for stock_code in untrained:
                    self._result_cache[stock_code] = list(results)  # 复制一份避免引用问题
                # 首次分析 → 显示 pending 提示
                pending_output = self.formatter.format_cached_fusion_pending(
                    code or "?", rule_decision, llm_decision, skills_decision
                )
                return pending_output, untrained
            # skip_cache=True → 正常显示融合（RL 显示为未训练）

        return output, untrained

    def _prompt_train_rl(self, codes: list):
        """询问用户是否对未训练的 RL 模型进行训练"""
        code_list = ", ".join(codes)
        print(f"  🎮 以下股票尚未训练 RL 模型: {code_list}")
        try:
            ans = input("  💡 是否现在训练？(y/n，默认 n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans not in ("y", "yes"):
            print("  ⏭️  跳过 RL 训练")
            print("  💡 后续可输入\"训练 <代码>\"或 CLI: stock-advisor train <代码>")
            # 用户拒绝训练 → 显示带 "RL 未训练" 的简化融合面板
            for code in codes:
                self._show_fallback_fusion(code)
            return

        for code in codes:
            print(f"\n  🧠 开始训练 {code} 的 RL 模型...")
            try:
                from src.tools.train_tool import train_model
                result = train_model(code=code)
                formatted = self.formatter.choose_formatter("train_model", result)
                print(formatted)

                if result.get("status") in ("ok", "updated", "already_fresh"):
                    # 训练成功 → 重新获取 RL 预测并展示完整融合
                    print(f"  🔄 正在重新生成 {code} 的三路融合结果...")
                    new_rl = self._get_rl_prediction_for_code(code)
                    self._display_cached_fusion(code, new_rl)
                else:
                    # 训练失败 → 显示简化融合
                    self._show_fallback_fusion(code)
            except Exception as e:
                print(f"  ❌ {code} 训练失败: {e}")
                self._show_fallback_fusion(code)
        print()

    def _show_fallback_fusion(self, code: str):
        """从缓存中取出结果，显示带 "RL 未训练" 的简化融合面板"""
        cached_results = self._result_cache.pop(code, None)
        if cached_results is None:
            return

        # 跳过缓存（避免循环），直接显示融合面板（RL 显示为"未训练"）
        fusion_output, _ = self._auto_fuse(cached_results, skip_cache=True)
        if fusion_output:
            print(fusion_output)

    def _get_rl_prediction_for_code(self, code: str) -> dict:
        """获取单只股票的 RL 预测（用于训练后重新预测）

        直接调用 rl_tool.get_rl_prediction，不经过 Executor/Planner。
        """
        from src.tools.rl_tool import get_rl_prediction
        return get_rl_prediction(code)

    def _display_cached_fusion(self, code: str, new_rl_result: dict):
        """从缓存中取出分析结果，替换 RL 结果后重新融合展示

        Args:
            code: 股票代码
            new_rl_result: 新的 RL 预测结果（已训练模型的推理结果）
        """
        cached_results = self._result_cache.pop(code, None)
        if cached_results is None:
            return

        # 替换缓存中的旧 RL 结果（untrained）为新结果
        updated_results = []
        for name, result in cached_results:
            if name == "get_rl_prediction":
                updated_results.append((name, new_rl_result))
            else:
                updated_results.append((name, result))

        # 重新生成融合面板（此时 RL 已训练，_auto_fuse 会走正常逻辑）
        fusion_output, _ = self._auto_fuse(updated_results)
        if fusion_output:
            print(fusion_output)
        else:
            # 后备：直接用 formatter 显示
            print(self.formatter.format_rl_result(new_rl_result))

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _read_input(self) -> Optional[str]:
        """读取用户输入，优先使用 prompt_toolkit"""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory
            from prompt_toolkit.styles import Style

            if not hasattr(self, '_pt_session'):
                style = Style.from_dict({
                    'prompt': 'bold green',
                })
                self._pt_session = PromptSession(
                    history=InMemoryHistory(),
                    style=style,
                )

            return self._pt_session.prompt([('class:prompt', '📊 > ')])

        except ImportError:
            return input("📊 > ")

    def _handle_command(self, text: str) -> bool:
        """处理内置命令，返回 True 表示已处理"""
        cmd = text.lower().strip()

        if cmd in ("/exit", "/quit", "/q", "exit", "quit", "q"):
            print("\n  再见!")
            self._running = False
            return True

        if cmd == "/help":
            print(self.HELP_TEXT)
            return True

        if cmd.startswith("/llm"):
            parts = text.strip().split()
            if len(parts) >= 2:
                sub = parts[1].lower()
                if sub == "on":
                    if not self._has_api:
                        print("  ⚠️  未设置 DEEPSEEK_API_KEY，无法开启 LLM")
                        print("      请设置环境变量后重启: DEEPSEEK_API_KEY=your_key")
                    else:
                        self.llm_enabled = True
                        self.executor.llm_enabled = True
                        self.planner.llm_enabled = True
                        print("  ✅ LLM 已开启，将使用 DeepSeek 深度推理")
                elif sub == "off":
                    self.llm_enabled = False
                    self.executor.llm_enabled = False
                    self.planner.llm_enabled = False
                    print("  🔒 LLM 已关闭，将使用本地增强策略引擎 (skills_analyze)")
                    print("     (11技能 LocalFusionEngine 5维评分卡 + 历史模式统计预测)")
                elif sub == "status":
                    if self.llm_enabled:
                        print(f"  🟢 LLM: 已开启 ({self.planner.model})")
                    elif self._has_api:
                        print("  🟡 LLM: 已手动关闭 (使用本地增强策略)")
                    else:
                        print("  🔴 LLM: 不可用 (未设置 API Key)")
                else:
                    print(f"  ❓ /llm {sub} — 可用: on / off / status")
            else:
                status = "🟢 已开启" if self.llm_enabled else ("🟡 已关闭" if self._has_api else "🔴 不可用")
                print(f"  LLM 状态: {status}")
                print(f"  用法: /llm on | off | status")
            return True

        if cmd == "/clear":
            self.memory.clear()
            print("  ✅ 会话记忆已清空")
            return True

        if cmd == "/history":
            ctx = self.memory.get_context_for_llm()
            if ctx:
                print(f"\n  📜 本轮记忆:\n{ctx}")
            else:
                print("  (暂无历史)")
            return True

        if cmd == "/tools":
            print("\n  🔧 可用工具:")
            for tool in ToolRegistry.list_all():
                print(f"    📎 {tool.name}: {tool.description[:80]}")
            print()
            return True

        if cmd == "/interested":
            from src.memory import get_interested_stocks
            mem = get_interested_stocks()
            if mem.count() == 0:
                print("  📭 还没有记录任何查询过的股票。")
            else:
                stocks = mem.top(10)
                print(f"\n  ⭐ 最常查询的股票 (Top 10):")
                for s in stocks:
                    print(f"    {s['code']} {s['name']:<8}  {s['count']:>2}次  {s['last_searched']}")
                print(f"    共 {mem.count()} 只 | data/interested_stocks.json")
            return True

        if cmd == "/debug":
            import os as _os
            debug_val = _os.getenv("STOCK_ADVISOR_DEBUG")
            if debug_val:
                del _os.environ["STOCK_ADVISOR_DEBUG"]
                print("  🐛 Debug 模式: OFF")
            else:
                _os.environ["STOCK_ADVISOR_DEBUG"] = "1"
                print("  🐛 Debug 模式: ON")
            return True

        return False


# ============================================================================
# 便捷入口
# ============================================================================

def main():
    """CLI 入口"""
    agent = AgentCore()
    agent.run()


if __name__ == "__main__":
    main()
