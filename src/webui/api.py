"""webui/api.py — REST API 路由

统一前缀 /api，全部 JSON。错误约定 {"error": "...", "code": "..."}。
"""

from flask import Blueprint, jsonify, request, Response, stream_with_context

from . import services
from .jobs import get_job_manager

api = Blueprint("api", __name__)


# ============================================================================
# 系统 / 市场
# ============================================================================

@api.route("/health")
def health():
    import time
    return jsonify({"status": "ok", "version": "0.1.0",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")})


@api.route("/market/state")
def market_state():
    return jsonify(services.get_market_state())


@api.route("/llm/status")
def llm_status():
    return jsonify(services.get_llm_status())


@api.route("/llm/config", methods=["GET"])
def llm_config_get():
    return jsonify(services.get_llm_config())


@api.route("/llm/config", methods=["PUT"])
def llm_config_put():
    data = request.get_json(silent=True) or {}
    params = data.get("config") or data
    return jsonify(services.save_llm_config(params))


@api.route("/llm/test", methods=["POST"])
def llm_test():
    data = request.get_json(silent=True) or {}
    params = data.get("config") or data
    return jsonify(services.test_llm_connection(params))


# ============================================================================
# 搜索
# ============================================================================

@api.route("/stocks/search")
def stock_search():
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 10))
    if not q:
        return jsonify({"results": []})
    return jsonify(services.search_stocks(q, limit=limit))


# ============================================================================
# K 线
# ============================================================================

@api.route("/stocks/<code>/kline")
def stock_kline(code):
    days = int(request.args.get("days", 500))
    return jsonify(services.get_kline(code, days=days))


# ============================================================================
# 实时行情
# ============================================================================

@api.route("/stocks/<code>/quote")
def stock_quote(code):
    return jsonify(services.get_quote(code))


# ============================================================================
# 分析 / 技能 / LLM / RL
# ============================================================================

@api.route("/stocks/<code>/analyze")
def stock_analyze(code):
    days = int(request.args.get("days", 500))
    return jsonify(services.get_analysis(code, days=days))


@api.route("/stocks/<code>/skills")
def stock_skills(code):
    return jsonify(services.get_skills_analysis(code))


@api.route("/stocks/<code>/llm")
def stock_llm(code):
    import os
    api_key = request.args.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
    return jsonify(services.get_llm_analysis(code, api_key=api_key))


@api.route("/stocks/<code>/rl")
def stock_rl(code):
    refresh = request.args.get("refresh", "0") == "1"
    return jsonify(services.get_rl_prediction(code, force_refresh=refresh))


# ============================================================================
# 历史 / 自选股
# ============================================================================

@api.route("/history")
def history():
    limit = int(request.args.get("limit", 50))
    return jsonify(services.get_history(limit=limit))


@api.route("/watchlist", methods=["GET"])
def watchlist_get():
    return jsonify(services.get_watchlist())


@api.route("/watchlist", methods=["POST"])
def watchlist_add():
    data = request.get_json(silent=True) or {}
    code = str(data.get("code", "")).strip()
    if not code or len(code) != 6 or not code.isdigit():
        return jsonify({"error": "无效的股票代码", "code": "bad_request"}), 400
    return jsonify(services.add_watchlist(code))


@api.route("/watchlist/<code>", methods=["DELETE"])
def watchlist_remove(code):
    return jsonify(services.remove_watchlist(code))


# ============================================================================
# 市场热门 / 列表
# ============================================================================

@api.route("/market/<market>/hot")
def market_hot(market):
    top = int(request.args.get("top", 20))
    return jsonify(services.get_market_hot(market, top=top))


@api.route("/market/<market>/stocks")
def market_list(market):
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 20))
    return jsonify(services.get_market_list(market, q=q, limit=limit))


# ============================================================================
# 规则参数编辑
# ============================================================================

@api.route("/config/rules", methods=["GET"])
def config_rules_get():
    return jsonify(services.get_rules_config())


@api.route("/config/rules", methods=["PUT"])
def config_rules_put():
    data = request.get_json(silent=True) or {}
    params = data.get("params") or data
    return jsonify(services.save_rules_config(params))


@api.route("/config/rules/reset", methods=["POST"])
def config_rules_reset():
    return jsonify(services.reset_rules_config())


@api.route("/config/skills", methods=["GET"])
def config_skills_get():
    return jsonify(services.get_skills_config())


@api.route("/config/skills", methods=["PUT"])
def config_skills_put():
    data = request.get_json(silent=True) or {}
    return jsonify(services.save_skills_config(data.get("params") or data))


# ============================================================================
# 异步 Job（RL 训练 / 预测 / 聊天）
# ============================================================================

@api.route("/jobs", methods=["POST"])
def submit_job():
    data = request.get_json(silent=True) or {}
    job_type = data.get("type")
    mgr = get_job_manager()

    if job_type == "train":
        code = str(data.get("code", "")).strip()
        timesteps = int(data.get("timesteps", 50000))
        if not code:
            return jsonify({"error": "缺少股票代码", "code": "bad_request"}), 400
        from src.webui.market_catalog import parse_code
        if parse_code(code)[0] != "a":
            return jsonify({"error": "RL 训练暂不支持港股/美股", "code": "bad_request"}), 400
        job_id, conflict_id = mgr.submit_train(code, timesteps)
        if job_id is None:
            return jsonify({"error": f"{code} 正在训练中", "code": "conflict",
                            "job_id": conflict_id}), 409
        return jsonify({"job_id": job_id, "type": "train", "code": code}), 202

    if job_type == "predict":
        codes = data.get("codes", [])
        use_llm = bool(data.get("use_llm", True))
        api_key = data.get("api_key")
        if not codes:
            return jsonify({"error": "缺少股票代码列表", "code": "bad_request"}), 400
        from src.webui.market_catalog import parse_code
        # 过滤非 A 股（三路融合仅支持 A 股）
        filtered = [c for c in codes if parse_code(str(c))[0] == "a"]
        if not filtered:
            return jsonify({"error": "三路融合暂不支持港股/美股", "code": "bad_request"}), 400
        job_id = mgr.submit_predict(filtered, use_llm=use_llm, api_key=api_key)
        return jsonify({"job_id": job_id, "type": "predict"}), 202

    if job_type == "chat":
        message = str(data.get("message", "")).strip()
        if not message:
            return jsonify({"error": "消息为空", "code": "bad_request"}), 400
        job_id = mgr.submit_chat(message)
        return jsonify({"job_id": job_id, "type": "chat"}), 202

    return jsonify({"error": f"未知任务类型: {job_type}", "code": "bad_request"}), 400


@api.route("/jobs/<job_id>")
def job_status(job_id):
    job = get_job_manager().get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在", "code": "not_found"}), 404
    return jsonify(job.to_dict())


@api.route("/jobs/<job_id>/stream")
def job_stream(job_id):
    def gen():
        yield from get_job_manager().stream_events(job_id)
    return Response(stream_with_context(gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@api.route("/jobs/<job_id>/cancel", methods=["POST"])
def job_cancel(job_id):
    ok = get_job_manager().cancel(job_id)
    if not ok:
        return jsonify({"error": "任务不存在", "code": "not_found"}), 404
    return jsonify({"ok": True})


@api.route("/jobs/active")
def jobs_active():
    return jsonify({"jobs": get_job_manager().active_jobs()})
