"""analytics/optimizer.py — AI 智能优化

自动微调策略参数、模型权重，适配市场风格切换。
"""

from typing import Dict, List, Optional, Callable
import numpy as np
from itertools import product


class StrategyOptimizer:
    """策略参数优化器

    使用网格搜索 + 模拟退火自动寻找最优参数组合。
    """

    def __init__(self, evaluate_fn: Callable):
        """
        Args:
            evaluate_fn: fn(params: dict) -> dict
                接收参数组合，返回绩效指标 {"sharpe": ..., "return": ..., "max_dd": ...}
        """
        self.evaluate = evaluate_fn
        self.best_params = None
        self.best_score = -float("inf")
        self.search_log = []

    def grid_search(self, param_grid: Dict[str, list], metric: str = "sharpe_ratio",
                    maximize: bool = True) -> Dict:
        """网格搜索最优参数

        Args:
            param_grid: {"param_name": [value1, value2, ...]}
            metric: 优化目标指标
            maximize: True 最大化 / False 最小化

        Returns:
            最优参数组合
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        best = None
        best_score = -float("inf") if maximize else float("inf")

        for combo in product(*values):
            params = dict(zip(keys, combo))
            try:
                result = self.evaluate(params)
                score = result.get(metric, 0)

                self.search_log.append({"params": params, "score": score})

                if maximize and score > best_score:
                    best_score = score
                    best = params
                elif not maximize and score < best_score:
                    best_score = score
                    best = params
            except Exception as e:
                continue

        self.best_params = best
        self.best_score = best_score
        return {"best_params": best, "best_score": best_score, "searched": len(self.search_log)}

    def random_search(self, param_dist: Dict[str, tuple], n_iter: int = 50,
                      metric: str = "sharpe_ratio", maximize: bool = True) -> Dict:
        """随机搜索（参数空间大时使用）

        Args:
            param_dist: {"param_name": (min, max)}
            n_iter: 迭代次数
        """
        best = None
        best_score = -float("inf") if maximize else float("inf")

        for _ in range(n_iter):
            params = {}
            for name, (lo, hi) in param_dist.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = np.random.randint(lo, hi + 1)
                else:
                    params[name] = np.random.uniform(lo, hi)

            try:
                result = self.evaluate(params)
                score = result.get(metric, 0)
                self.search_log.append({"params": params, "score": score})
                if maximize and score > best_score:
                    best_score = score
                    best = params
                elif not maximize and score < best_score:
                    best_score = score
                    best = params
            except Exception:
                continue

        return {"best_params": best, "best_score": best_score, "searched": n_iter}


class FusionWeightOptimizer:
    """决策融合权重优化器

    根据近期表现自动调整 RL 与规则的融合权重。
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.performance_log: List[dict] = []

    def update(self, rl_accuracy: float, rule_accuracy: float):
        """记录近期准确率"""
        self.performance_log.append({"rl": rl_accuracy, "rule": rule_accuracy})
        if len(self.performance_log) > self.window * 2:
            self.performance_log.pop(0)

    def optimize_weights(self) -> Dict[str, float]:
        """基于近期准确率优化权重"""
        if len(self.performance_log) < self.window:
            return {"rl_weight": 0.6, "rule_weight": 0.4}

        recent = self.performance_log[-self.window:]
        rl_avg = np.mean([r["rl"] for r in recent])
        rule_avg = np.mean([r["rule"] for r in recent])
        total = rl_avg + rule_avg

        if total == 0:
            return {"rl_weight": 0.5, "rule_weight": 0.5}

        return {
            "rl_weight": round(rl_avg / total, 4),
            "rule_weight": round(rule_avg / total, 4),
        }
