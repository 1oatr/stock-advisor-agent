"""webui/chat.py — 聊天服务

AgentCore 单例（会话记忆跨请求持久）+ stdout 捕获（把 query() 中间的
print 逐行推给 Job 的 log 事件，最终返回值作 result 事件）。
"""

import contextlib
import threading

_agent = None
_agent_lock = threading.Lock()


class _PusherIO:
    """把 write 的内容按行入队到 job 的 log 事件"""

    def __init__(self, job):
        self.job = job
        self._buf = ""

    def write(self, s: str):
        if not s:
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self.job.log(line)

    def flush(self):
        if self._buf.strip():
            self.job.log(self._buf.strip())
            self._buf = ""


def _get_agent():
    """获取 AgentCore 单例（线程安全）

    首次创建时应用 data/llm_config.json 的配置（api_key/api_base/model/开关）。
    """
    global _agent
    with _agent_lock:
        if _agent is None:
            from src.webui import services
            cfg = services.get_llm_config()
            from src.agent.core import AgentCore
            _agent = AgentCore(
                api_key=cfg.get("api_key") or None,
                api_base=cfg.get("api_base"),
                model=cfg.get("model"),
            )
            # LLM 开关：配置决定（未显式保存过 enabled 时 cfg 默认=有无 key，
            # 与 AgentCore 内部逻辑一致，强制覆盖无副作用）
            _agent.llm_enabled = bool(cfg.get("enabled"))
            _agent.executor.llm_enabled = _agent.llm_enabled
        return _agent


def reset_agent():
    """LLM 配置变更后调用：重置单例，下次聊天用新配置重建"""
    global _agent
    with _agent_lock:
        _agent = None


def run_chat(message: str, job):
    """在后台线程中执行聊天，中间输出和结果推给 job。"""
    agent = _get_agent()

    # 聊天锁：保证 SessionMemory 与 stdout 重定向线程安全
    with _agent_lock:
        job.update_progress(0.1, "Agent 调度中...")
        pusher = _PusherIO(job)
        with contextlib.redirect_stdout(pusher):
            reply = agent.query(message)
        job.update_progress(0.9, "生成完成")
        job.set_result({"reply": reply})
