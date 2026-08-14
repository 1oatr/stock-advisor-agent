"""rl/agent.py — 单股 RL 推理智能体

加载训练好的单股模型，对当前行情做预测。
支持模型缓存、新鲜度检查、以及未训练时的安全回退。
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from .single_env import SingleStockEnv
from .train import load_single_model, check_model_freshness

logger = logging.getLogger(__name__)


class SingleStockAgent:
    """单股 RL 推理智能体

    用法:
        agent = SingleStockAgent()
        result = agent.predict(df, "600519")
        # → {"action": "buy", "confidence": 0.72, "untrained": False}
    """

    def __init__(self, model_dir: str = "models", cache_models: bool = True,
                 window: int = 60):
        self.model_dir = model_dir
        self.cache_models = cache_models
        self.window = window
        self._model_cache: Dict[str, any] = {}         # code → PPO model
        self._freshness_cache: Dict[str, Dict] = {}    # code → freshness info

        os.makedirs(model_dir, exist_ok=True)

    # ========================================================================
    # 推理 API
    # ========================================================================

    def predict(self, df: pd.DataFrame, code: str) -> Dict:
        """对单只股票进行 RL 推理

        Args:
            df: 历史数据 (需含技术指标)
            code: 股票代码

        Returns:
            {
                "action": "buy"|"sell"|"hold",
                "confidence": 0.72,
                "untrained": False,
                "model_fresh": True,
                "days_since_train": 3,
            }
        """
        if df.empty or len(df) < 30:
            return self._fallback("数据不足")

        # 检查/加载模型
        model = self._get_model(code)
        if model is None:
            return self._fallback("未训练")

        # 检查新鲜度
        freshness = self._get_freshness(code)

        # 创建环境并运行推理（eval_mode 确定性起始）
        try:
            env = SingleStockEnv(df, code=code, window=self.window)
            obs, _ = env.reset(options={"eval_mode": True})

            # 运行完整 episode 获取最终动作
            action, confidence, details = self._run_episode(model, env)

            return {
                "action": action,
                "confidence": round(confidence, 4),
                "untrained": False,
                "model_fresh": freshness.get("is_fresh", False),
                "days_since_train": freshness.get("days_since_train"),
                "eval_return": freshness.get("eval_return"),
                "details": details,
            }

        except Exception as e:
            logger.warning(f"{code}: RL 推理异常 → {e}")
            return self._fallback(f"推理异常: {e}")

    def predict_with_indicators(self, indicators_snapshot: dict, code: str) -> Dict:
        """基于指标快照做快速预测（轻量版，不需要完整 df）

        如果模型未训练，返回 fallback。
        """
        model = self._get_model(code)
        if model is None:
            return self._fallback("未训练")

        # 通过模型直接判断（简化版）
        freshness = self._get_freshness(code)
        return {
            "action": "hold",
            "confidence": 0.50,
            "untrained": False,
            "model_fresh": freshness.get("is_fresh", False),
            "note": "指标快照模式（简化推理）",
        }

    def is_trained(self, code: str) -> bool:
        """检查某股票是否已训练模型"""
        model_path = os.path.join(self.model_dir, f"{code}_ppo.zip")
        return os.path.exists(model_path)

    def clear_cache(self):
        """清空模型缓存"""
        self._model_cache.clear()
        self._freshness_cache.clear()

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _get_model(self, code: str):
        """获取模型（带缓存）"""
        if self.cache_models and code in self._model_cache:
            return self._model_cache[code]

        model = load_single_model(code, self.model_dir)
        if model and self.cache_models:
            self._model_cache[code] = model
        return model

    def _get_freshness(self, code: str) -> Dict:
        """获取新鲜度（带缓存）"""
        if code not in self._freshness_cache:
            self._freshness_cache[code] = check_model_freshness(code, self.model_dir)
        return self._freshness_cache[code]

    def _run_episode(self, model, env: SingleStockEnv) -> tuple:
        """运行推理 episode

        Returns:
            (action, confidence, details_dict)
        """
        obs, _ = env.reset()
        done = False
        actions_taken = []
        action_probs = []

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # 获取动作概率（如果模型支持）
            try:
                probs = model.policy.get_distribution(obs).distribution.probs
                if probs is not None:
                    prob = float(probs.detach().cpu().numpy()[0][action])
                    action_probs.append(prob)
            except Exception:
                pass

            obs, reward, terminated, truncated, _ = env.step(action)
            actions_taken.append(int(action))
            done = terminated or truncated

        # 决策：以 episode 中第一次买入/卖出动作为主
        buy_count = actions_taken.count(1)
        sell_count = actions_taken.count(2)
        hold_count = actions_taken.count(0)

        total = len(actions_taken)
        avg_prob = np.mean(action_probs) if action_probs else 0.65

        # 确定主方向
        if buy_count > sell_count and buy_count > hold_count:
            action = "buy"
            confidence = min(buy_count / total * 0.9 + avg_prob * 0.3, 0.95)
        elif sell_count > buy_count and sell_count > hold_count:
            action = "sell"
            confidence = min(sell_count / total * 0.9 + avg_prob * 0.3, 0.95)
        else:
            action = "hold"
            # 持有置信度 = 持有比例 + 模型不确定性
            confidence = hold_count / total * 0.6 + 0.25

        details = {
            "buy_ratio": round(buy_count / total, 3),
            "sell_ratio": round(sell_count / total, 3),
            "hold_ratio": round(hold_count / total, 3),
            "total_steps": total,
        }

        return action, confidence, details

    def _fallback(self, reason: str = "") -> Dict:
        """模型不可用时的安全回退"""
        return {
            "action": "hold",
            "confidence": 0.50,
            "untrained": True,
            "reason": reason,
        }
