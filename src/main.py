"""main.py — Stock Advisor CLI 入口

两用模式：
    stock-advisor              → 启动交互式 REPL（Agent 自主调度）
    stock-advisor <command>    → 直接执行子命令（兼容旧版）

三方决策: LLM+Skills(40%) + 硬规则(30%) + RL智能体(30%)
"""

import sys
import os
from pathlib import Path

# 确保能找到 src 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 修复 Windows GBK 终端下的 emoji/中文输出乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import click
import pandas as pd
from datetime import datetime


# ============================================================================
# 全局常量
# ============================================================================


# ============================================================================
# 命令: repl — 交互式 Agent REPL（默认）
# ============================================================================

@click.command()
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", help="DeepSeek API Key")
@click.option("--api-base", default="https://api.deepseek.com", help="API Base URL")
@click.option("--model", default="deepseek-v4-flash", help="模型名称")
@click.option("--no-color", is_flag=True, help="禁用 rich 彩色输出")
def repl(api_key, api_base, model, no_color):
    """🤖 启动交互式 Agent REPL（默认模式）

    输入自然语言，Agent 自动调度分析模块。
    """
    # 设置编码
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    from src.agent.core import AgentCore

    agent = AgentCore(
        api_key=api_key,
        api_base=api_base,
        model=model,
    )
    agent.run()
    sys.exit(0)  # 确保正常退出时返回码为 0


# ============================================================================
# 命令: scan — 扫描热门股票
# ============================================================================

@click.command()
@click.option("--top", default=20, help="筛选热门股数量")
@click.option("--save", is_flag=True, help="保存结果到文件")
def scan(top, save):
    """📡 扫描当前热门股票并排名"""
    from src.scanner.scanner import scan_hot_stocks
    from src.scanner.market_watch import MarketWatch
    from src.data.fetcher import DataFetcher

    fetcher = DataFetcher()
    market = MarketWatch(fetcher=fetcher)
    market_state = market.update()
    state_icon = {"bull": "🚀", "bear": "📉", "range": "📊", "unknown": "❓"}

    click.echo(f"📡 扫描热门股票 Top {top}")
    click.echo(f"  大盘: {state_icon.get(market_state['state'], '❓')} "
               f"{market_state['state']}  "
               f"(20日: {market_state.get('ret_20d', 0):+.1f}%)")

    results = scan_hot_stocks(top_n=top, fetcher=fetcher)
    if not results:
        click.echo("  ❌ 未能获取到热门股票数据")
        return

    click.echo(f"\n{'='*60}")
    click.echo(f"🔥 热门股票 TOP {len(results)}")
    click.echo(f"{'='*60}")

    for i, r in enumerate(results, 1):
        click.echo(f"  #{i:2d} {r['code']:6s} {r.get('name', ''):8s}  "
                   f"分数:{r['score']:6.2f}  收盘:{r.get('last_close', 0):>8.2f}  "
                   f"量比:{r.get('volume_ratio', 0):.1f}")

    click.echo(f"{'='*60}")

    if save:
        out_path = f"data/scan_{datetime.now():%Y%m%d_%H%M}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        click.echo(f"  💾 已保存: {out_path}")


# ============================================================================
# 命令: analyze — 深度分析单只股票
# ============================================================================

@click.command()
@click.argument("code")
@click.option("--days", default=120, help="回看天数")
def analyze(code, days):
    """🔍 深度技术分析单只股票"""
    from src.tools.analyze_tool import analyze_stock

    click.echo(f"🔍 分析 {code}（回看 {days} 天）...\n")
    result = analyze_stock(code, days=days)

    if "error" in result:
        click.echo(f"  ❌ {result['error']}")
        return

    # 简洁输出
    ind = result["indicators"]
    rule = result["rule_engine"]
    skills = result["skills"]

    click.echo(f"═══ {code} 技术分析 ═══")
    click.echo(f"  收盘: {result['last_close']:.2f}  |  区间: {result['period_return']:+.1f}%")
    click.echo(f"  MA5={ind.get('MA5','?')} MA20={ind.get('MA20','?')} MA60={ind.get('MA60','?')}")
    click.echo(f"  RSI={ind.get('RSI','?')} | MACD_DIF={ind.get('MACD_DIF','?')}")
    click.echo(f"  布林%B={ind.get('BOLL_POSITION','?')} | 量比={ind.get('VOL_RATIO','?')}")
    click.echo(f"")
    click.echo(f"  📐 规则: {rule['composite_signal']} (强度{rule['composite_strength']:.0%})")
    for r in rule["top_rules"][:3]:
        icon = {"buy": "✅", "sell": "❌", "hold": "➖"}
        click.echo(f"    {icon.get(r['signal'],'➖')} {r['name']}: {r.get('explanation','')}")
    click.echo(f"  📊 技能: {skills['aggregate_signal']} (置信度{skills['confidence']:.0%})")


# ============================================================================
# 命令: predict — 三路融合预测
# ============================================================================

@click.command()
@click.argument("codes", nargs=-1, required=True)
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", help="DeepSeek API Key")
@click.option("--no-llm", is_flag=True, help="关闭 LLM，使用本地增强策略引擎")
def predict(codes, api_key, no_llm):
    """📊 三路融合预测（LLM+RL+规则）"""
    from src.tools.predict_tool import predict_stocks
    from src.agent.formatter import Formatter
    fmt = Formatter()

    stock_list = list(codes)
    llm_status = "🔒 本地策略" if no_llm else "🧠 LLM"
    click.echo(f"📊 三路融合预测 ({llm_status}): {', '.join(stock_list)}\n")

    result = predict_stocks(stock_list, api_key=api_key, use_llm=not no_llm)

    state_icon = {"bull": "🚀", "bear": "📉", "range": "📊"}
    click.echo(f"  大盘: {state_icon.get(result['market_state'], '❓')} {result['market_state']}")

    for rec in result.get("recommendations", []):
        if "error" in rec:
            click.echo(f"  ❌ {rec['code']}: {rec['error']}")
            continue

        fused = rec["fused"]
        llm = rec["llm_skills"]
        rules = rec["rules"]
        rl = rec["rl"]
        a_icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}

        click.echo(f"\n  ┌─ {rec['code']} ─────────────────────┐")
        click.echo(f"  │ 🧠 LLM+Skills: {a_icon.get(llm['action'],'⚪')} {fmt._full_advice(llm['action'], llm['confidence'])}")
        click.echo(f"  │ 📐 硬规则:    {a_icon.get(rules['signal'],'⚪')} {fmt._full_advice(rules['signal'], rules['strength'])}")
        if rl.get("untrained"):
            rl_display = "未训练，无法给出建议"
        else:
            rl_display = f"{a_icon.get(rl['action'],'⚪')} {fmt._full_advice(rl['action'], rl['confidence'])}"
        click.echo(f"  │ 🎮 RL智能体:  {rl_display}")
        click.echo(f"  ├─ 🔗 最终建议: {a_icon.get(fused['action'],'⚪')} {fmt._full_advice(fused['action'], fused['confidence'], fused['position'])}")
        click.echo(f"  └──────────────────────────────────┘")

    click.echo("")


# ============================================================================
# 命令: train — RL 模型训练
# ============================================================================

@click.command()
@click.argument("code")
@click.option("--timesteps", default=200000, type=int, help="训练步数")
@click.option("--force", is_flag=True, help="强制全量训练")
def train(code, timesteps, force):
    """🧠 训练单股 RL 交易模型"""
    from src.data.fetcher import DataFetcher
    from src.data.indicators import add_all_indicators
    from src.data.cleaning import DataCleaner
    from src.data.enrichment import enrich_all
    from src.rl.train import train_single_stock

    click.echo(f"🧠 训练 RL 模型: {code} ({timesteps:,} 步)")

    fetcher = DataFetcher()
    cleaner = DataCleaner()

    from datetime import timedelta
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    # 1. 个股日线
    df = fetcher.daily_bars(code, start=start, end="")
    if df.empty:
        click.echo(f"  ❌ {code}: 无数据")
        return

    # 2. 大盘指数
    index_df = None
    try:
        idx_code = fetcher.market_index_for_stock(code)
        index_df = fetcher.index_daily(idx_code, start=start)
        click.echo(f"  📥 大盘指数: {idx_code} ({len(index_df)} 行)")
    except Exception as e:
        click.echo(f"  ⚠️ 大盘数据: {e}（用 0 填充）")

    # 3. 资金流向
    fund_flow_df = None
    try:
        fund_flow_df = fetcher.fund_flow(code)
        click.echo(f"  📥 资金流向: {len(fund_flow_df)} 行")
    except Exception as e:
        click.echo(f"  ⚠️ 资金流向: {e}（用 0 填充）")

    # 4. 清洗 + 增强 + 指标
    df = cleaner.clean_single(df, code)
    df = enrich_all(df, code, index_df=index_df, fund_flow_df=fund_flow_df)
    df = add_all_indicators(df)
    click.echo(f"  📥 总数据: {len(df)} 行, {len(df.columns)} 列")

    result = train_single_stock(df, code, timesteps=timesteps, force_full=force)

    if result.get("status") == "error":
        click.echo(f"  ❌ {result.get('message')}")
        return

    metrics = result.get("eval_metrics", {})
    click.echo(f"\n  ✅ 训练完成 ({result['mode']})")
    click.echo(f"  总收益: {metrics.get('total_return_pct', 0):+.1f}%")
    click.echo(f"  夏普比: {metrics.get('sharpe', 0)}")
    click.echo(f"  胜率:   {metrics.get('win_rate', 0)}%")
    click.echo(f"  交易:   {metrics.get('trades', 0)} 笔")
    click.echo(f"  模型:   {result.get('model_path', '?')}")


# ============================================================================
# 命令: backtest — 回测
# ============================================================================

@click.command()
@click.option("--strategy", default="trend_following", help="策略名称")
@click.option("--codes", default="000001,000002,000858", help="股票代码（逗号分隔）")
@click.option("--start", default="2024-01-01", help="起始日期")
@click.option("--end", default="2025-06-30", help="截止日期")
@click.option("--cash", default=1000000, type=int, help="初始资金")
def backtest(strategy, codes, start, end, cash):
    """🔬 历史回测"""
    from src.data.fetcher import DataFetcher
    from src.data.indicators import add_all_indicators
    from src.backtest.engine import BacktestEngine
    from src.strategy.templates import get_strategy

    stock_codes = [c.strip() for c in codes.split(",")]
    click.echo(f"🔬 回测 [{strategy}] {start}~{end}  标的:{stock_codes}")

    fetcher = DataFetcher()
    data = {}
    for code in stock_codes:
        df = fetcher.daily_bars(code, start=start, end=end)
        if not df.empty:
            df = add_all_indicators(df)
            data[code] = df.set_index("date")

    if not data:
        click.echo("  ❌ 无数据")
        return

    strat = get_strategy(strategy)

    def signal_fn(date, positions, cash, day_data):
        return strat.next(date, day_data, positions, cash)

    engine = BacktestEngine(initial_cash=cash)
    engine.t_plus_1 = True
    metrics = engine.run(data, signal_fn)

    click.echo(f"\n{'='*50}")
    click.echo(f"📊 回测结果")
    click.echo(f"  总收益: {metrics.get('total_return', 0):+.2f}%")
    click.echo(f"  年化:   {metrics.get('annual_return', 0):+.2f}%")
    click.echo(f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
    click.echo(f"  夏普:   {metrics.get('sharpe_ratio', 0):.2f}")
    click.echo(f"  胜率:   {metrics.get('win_rate', 0):.1f}%")
    click.echo(f"{'='*50}")


# ============================================================================
# 命令: compare — 多股对比
# ============================================================================

@click.command()
@click.argument("codes", nargs=-1, required=True)
def compare(codes):
    """📊 多股技术面对比"""
    from src.tools.compare_tool import compare_stocks

    stock_list = list(codes)
    click.echo(f"📊 对比: {', '.join(stock_list)}\n")

    result = compare_stocks(stock_list)
    table = result.get("comparison_table", [])

    for r in table:
        if "error" in r:
            click.echo(f"  ❌ {r['code']}: {r['error']}")
            continue
        icon = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}
        click.echo(f"  {icon.get(r['rule_signal'],'⚪')} {r['code']:6s} "
                   f"收盘:{r['close']:>8.2f}  涨跌:{r['period_return_pct']:>+6.1f}%  "
                   f"RSI={r['RSI']:>5.1f}  规则={r['rule_signal']}({r['rule_strength']:.0%})")

    verdict = result.get("verdict", {})
    if verdict.get("best"):
        click.echo(f"\n  🏆 最强: {verdict['best']}")
    if verdict.get("worst"):
        click.echo(f"  ⚠️  最弱: {verdict['worst']}")


# ============================================================================
# 命令: skills — 技能管理
# ============================================================================

@click.command()
@click.option("--list", "list_skills", is_flag=True, help="列出所有可用技能")
@click.argument("code", required=False)
def skills(list_skills, code):
    """📊 决策技能管理"""
    from src.skills import init_skills, get_manager, SkillRegistry

    init_skills()

    if list_skills:
        click.echo("📊 可用决策技能:")
        for cat, skills in SkillRegistry.list_by_category().items():
            click.echo(f"\n  📂 {cat}:")
            for s in skills:
                click.echo(f"    ✅ {s.name}: {s.description}")
        click.echo(f"\n  总计: {SkillRegistry.count()} 个技能")

    elif code:
        from src.data.fetcher import DataFetcher
        from src.data.indicators import add_all_indicators
        from src.data.cleaning import DataCleaner

        fetcher = DataFetcher()
        df = fetcher.daily_bars(code, start="", end="")
        if df.empty:
            click.echo(f"  ❌ {code}: 无数据")
            return

        df = DataCleaner().clean_single(df, code)
        df = add_all_indicators(df)
        mgr = get_manager()
        results = mgr.run_all(df.tail(120), code)
        agg = mgr.aggregate_signal(results)
        click.echo(mgr.format_results(results))
        click.echo(f"\n  综合: {agg['signal']} (置信度{agg['confidence']:.0%})")

    else:
        click.echo(f"📊 已注册 {SkillRegistry.count()} 个技能")
        click.echo(f"  stock-advisor skills --list    列出所有")
        click.echo(f"  stock-advisor skills 000001    分析股票")


@click.command()
@click.option("--list", "list_stocks", is_flag=True, help="列出所有感兴趣股票")
@click.option("--top", default=10, help="按热度显示前 N 只")
@click.option("--recent", is_flag=True, help="按最近查询时间排序")
def interested(list_stocks, top, recent):
    """⭐ 查看感兴趣股票（长期记忆）"""
    from src.memory import get_interested_stocks

    mem = get_interested_stocks()

    if mem.count() == 0:
        click.echo("📭 还没有记录任何查询过的股票。")
        click.echo("   试试 stock-advisor analyze <代码> 开始分析！")
        return

    if recent:
        stocks = mem.all()[:top]
        title = f"⭐ 近期查询过的股票 (Top {len(stocks)})"
    else:
        stocks = mem.top(top)
        title = f"⭐ 最常查询的股票 (Top {len(stocks)})"

    click.echo(f"\n{title}")
    click.echo(f"{'='*55}")
    click.echo(f"{'代码':<10} {'名称':<10} {'次数':<6} {'最近查询':<20}")
    click.echo(f"{'-'*55}")
    for s in stocks:
        click.echo(
            f"{s['code']:<10} {s['name']:<10} {s['count']:<6} {s['last_searched']:<20}"
        )
    click.echo(f"{'='*55}")
    click.echo(f"  共 {mem.count()} 只股票")
    click.echo(f"  数据文件: data/interested_stocks.json")


# ============================================================================
# CLI 入口
# ============================================================================

@click.group(invoke_without_command=True)
@click.option("--api-key", envvar="DEEPSEEK_API_KEY", help="DeepSeek API Key")
@click.option("--model", default="deepseek-v4-flash", help="LLM 模型")
@click.pass_context
def cli(ctx, api_key, model):
    """📊 Stock Advisor — 全栈 A 股量化分析系统

    LLM+Skills(40%) + 硬规则(30%) + RL智能体(30%) 三方融合决策。

    直接运行启动交互式 REPL。
    """
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["model"] = model

    # 无子命令 → 启动 REPL
    if ctx.invoked_subcommand is None:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        from src.agent.core import AgentCore
        agent = AgentCore(api_key=api_key, model=model)
        agent.run()
        sys.exit(0)  # 确保正常退出时返回码为 0


# 注册子命令
cli.add_command(repl, name="repl")
cli.add_command(scan, name="scan")
cli.add_command(analyze, name="analyze")
cli.add_command(predict, name="predict")
cli.add_command(train, name="train")
cli.add_command(backtest, name="backtest")
cli.add_command(compare, name="compare")
cli.add_command(skills, name="skills")
cli.add_command(interested, name="interested")


if __name__ == "__main__":
    cli()
