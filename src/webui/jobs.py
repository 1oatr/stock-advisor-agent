"""webui/jobs.py — 异步任务队列

管理长任务（RL 训练 / 批量预测 / 聊天）：
- Job 状态机: queued → running → done | error | cancelled
- daemon 线程执行，不阻塞 Flask
- SSE 事件流（progress / log / result / error / done，15s 心跳）
- 防重复训练: running_by_code 映射，同代码 409
"""

import threading
import time
import uuid
from collections import deque
from typing import Optional


class Job:
    def __init__(self, job_type: str, params: dict, code: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.type = job_type
        self.params = params
        self.code = code
        self.status = "queued"
        self.progress = 0.0
        self.message = "排队中..."
        self.result = None
        self.error = None
        self.cancelled = False
        self.cancel_event = threading.Event()
        self.created_at = time.time()
        self._events = deque()
        self._lock = threading.Lock()

    def emit(self, event: str, data: dict):
        with self._lock:
            self._events.append({"event": event, "data": data})

    def update_progress(self, frac: float, message: str = ""):
        frac = max(0.0, min(1.0, frac))
        with self._lock:
            self.progress = frac
            if message:
                self.message = message
        self.emit("progress", {"progress": frac, "message": message})

    def log(self, line: str):
        self.emit("log", {"line": line})

    def set_result(self, result):
        self.result = result
        self.status = "done"
        self.progress = 1.0
        self.emit("done", {"job_id": self.id})
        self.emit("result", {"result": result})

    def set_error(self, error: str):
        self.error = error
        self.status = "error"
        self.emit("error", {"error": error})

    def mark_cancelled(self):
        self.cancelled = True
        self.status = "cancelled"
        self.emit("done", {"job_id": self.id, "cancelled": True})

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "type": self.type,
                "code": self.code,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "result": self.result,
                "error": self.error,
                "cancelled": self.cancelled,
            }

    def pending_events(self):
        with self._lock:
            out = list(self._events)
            self._events.clear()
            return out


class JobManager:
    def __init__(self):
        self._jobs = {}
        self._running_by_code = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 提交任务
    # ------------------------------------------------------------------

    def submit_train(self, code: str, timesteps: int = 50000) -> tuple:
        """提交 RL 训练。同代码已有训练任务时返回 (None, 冲突 job_id)。"""
        with self._lock:
            if code in self._running_by_code:
                return None, self._running_by_code[code]

            job = Job("train", {"code": code, "timesteps": timesteps}, code=code)
            job.status = "running"
            job.message = "准备数据..."
            job.emit("progress", {"progress": 0.0, "message": "准备数据..."})
            self._jobs[job.id] = job
            self._running_by_code[code] = job.id

            t = threading.Thread(
                target=self._run_train, args=(job, code, timesteps), daemon=True
            )
            t.start()
            return job.id, None

    def submit_predict(self, codes: list, use_llm: bool = True,
                       api_key: Optional[str] = None) -> str:
        job = Job("predict", {"codes": codes, "use_llm": use_llm})
        job.status = "running"
        job.message = "开始三路融合分析..."
        self._jobs[job.id] = job
        threading.Thread(
            target=self._run_predict, args=(job, codes, use_llm, api_key), daemon=True
        ).start()
        return job.id

    def submit_chat(self, message: str) -> str:
        job = Job("chat", {"message": message})
        job.status = "running"
        job.message = "分析中..."
        self._jobs[job.id] = job
        threading.Thread(
            target=self._run_chat, args=(job, message), daemon=True
        ).start()
        return job.id

    # ------------------------------------------------------------------
    # 查询 / 取消 / 事件流
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        job.message = "正在取消..."
        return True

    def active_jobs(self) -> list:
        with self._lock:
            return [
                {"id": j.id, "type": j.type, "code": j.code,
                 "status": j.status, "progress": j.progress, "message": j.message}
                for j in self._jobs.values()
                if j.status in ("queued", "running")
            ]

    def stream_events(self, job_id: str):
        """SSE 生成器：重放历史事件，之后阻塞读新事件，15s 心跳。"""
        job = self._jobs.get(job_id)
        if job is None:
            yield "event: error\ndata: {\"error\": \"job not found\"}\n\n"
            return

        # 重放
        for ev in job.pending_events():
            yield f"event: {ev['event']}\ndata: {__import__('json').dumps(ev['data'], ensure_ascii=False)}\n\n"

        # 已结束则直接返回
        if job.status in ("done", "error", "cancelled"):
            return

        # 阻塞读新事件，15s 心跳保活
        while job.status in ("queued", "running"):
            for ev in job.pending_events():
                yield f"event: {ev['event']}\ndata: {__import__('json').dumps(ev['data'], ensure_ascii=False)}\n\n"
            time.sleep(0.3)
            # 心跳（简化：每 ~15s 发一次）
            yield ": ping\n\n"

        # 结束事件
        for ev in job.pending_events():
            yield f"event: {ev['event']}\ndata: {__import__('json').dumps(ev['data'], ensure_ascii=False)}\n\n"

    # ------------------------------------------------------------------
    # 任务实现
    # ------------------------------------------------------------------

    def _run_train(self, job: Job, code: str, timesteps: int):
        import traceback

        # 训练阶段进度: 0-5% 拉数据, 5-85% learn, 85-95% 评估, 95-100% 保存
        from stable_baselines3.common.callbacks import BaseCallback

        class TrainProgressCallback(BaseCallback):
            def __init__(self, job, total, train_lo=0.05, train_hi=0.85):
                super().__init__(verbose=0)
                self.job = job
                self.total = max(total, 1)
                self.lo, self.hi = train_lo, train_hi

            def _on_step(self) -> bool:
                frac = min(self.num_timesteps / self.total, 1.0)
                pct = self.lo + (self.hi - self.lo) * frac
                self.job.update_progress(
                    pct, f"训练中 {self.num_timesteps:,}/{self.total:,} 步 ({frac*100:.1f}%)"
                )
                if self.job.cancel_event.is_set():
                    self.job.message = "正在停止(用户取消)..."
                    return False
                return True

        try:
            job.update_progress(0.02, "获取数据 + 清洗 + 指标计算...")

            from src.tools.train_tool import train_model
            progress_cb = TrainProgressCallback(job, timesteps)

            def cancel_callback():
                return job.cancel_event.is_set()

            from src.webui import services
            result = train_model(
                code=code, timesteps=timesteps,
                extra_callbacks=[progress_cb],
                cancel_event=cancel_callback,
                config=services.rl_train_config(),
            )

            if job.cancel_event.is_set():
                job.message = "训练已取消（未完整训练，未保存模型）"
                job.mark_cancelled()
                # 清理 running_by_code
                with self._lock:
                    self._running_by_code.pop(code, None)
                return

            job.update_progress(0.95, "保存模型...")
            # 训练完成 → 清除该股票的 RL/分析/技能缓存，前端刷新才能拿到新结果
            from src.webui import services
            services.clear_stock_cache(code)
            job.set_result(result)
        except Exception as e:
            traceback.print_exc()
            job.set_error(f"训练失败: {e}")
        finally:
            with self._lock:
                self._running_by_code.pop(code, None)

    def _run_predict(self, job: Job, codes: list, use_llm: bool, api_key):
        from src.webui import services
        try:
            job.update_progress(0.1, f"分析 {len(codes)} 只股票...")
            result = services.predict_stocks(codes, use_llm=use_llm, api_key=api_key)
            job.update_progress(0.95, "融合完成")
            job.set_result(result)
        except Exception as e:
            job.set_error(f"预测失败: {e}")

    def _run_chat(self, job: Job, message: str):
        from src.webui import chat
        try:
            chat.run_chat(message, job)
        except Exception as e:
            job.set_error(f"聊天失败: {e}")


# 全局单例
_manager = JobManager()


def get_job_manager() -> JobManager:
    return _manager
