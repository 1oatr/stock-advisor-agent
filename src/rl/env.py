"""rl/env.py — 自定义多股交易环境

Gymnasium 环境，支持多只股票并行交易。
状态 = N 股 × M 特征;  动作 = 每只股票的 [卖出, 持有, 买入] + 仓位比例
"""

from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class MultiStockTradingEnv(gym.Env):
    """多股票交易环境

    Observation:
        Type: Box(shape=(window, n_stocks * n_features))
        过去 window 天的多股特征矩阵

    Action:
        Type: Box(shape=(n_stocks, 2))
        action[:, 0]: -1 (卖出) ~ 1 (买入), 0 持有
        action[:, 1]: 仓位比例 0 ~ 1

    Reward:
        组合收益率 - 风险惩罚 (夏普比率导向)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stock_codes: list,
        feature_cols: list,
        window: int = 60,
        transaction_cost: float = 0.0003,
        max_position: float = 0.2,
    ):
        super().__init__()

        self.n_stocks = len(stock_codes)
        self.n_features = len(feature_cols)
        self.window = window
        self.stock_codes = stock_codes
        self.feature_cols = feature_cols
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        # 预处理数据
        self._preprocess_data(df)

        # 观测空间: (window, n_stocks * n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window, self.n_stocks * self.n_features),
            dtype=np.float32,
        )

        # 动作空间: 每只股票 [direction, position_ratio]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_stocks, 2), dtype=np.float32
        )

        # 状态变量
        self.current_step = window
        self.portfolio_value = 1.0
        self.positions = np.zeros(self.n_stocks, dtype=np.float32)
        self.cash = 1.0
        self.max_steps = len(self._dates) - window - 1
        self._returns_history = []

    def _preprocess_data(self, df: pd.DataFrame):
        """将多股数据转换为时间序列矩阵

        生成:
            self._prices: (total_steps, n_stocks) 每只股票的价格序列
            self._features: (total_steps, n_stocks, n_features) 特征张量
            self._dates: 交易日列表
        """
        # 获取所有交易日并排序
        all_dates = df["date"].unique() if "date" in df.columns else df.index
        self._dates = sorted(pd.to_datetime(all_dates))

        n_dates = len(self._dates)
        self._prices = np.zeros((n_dates, self.n_stocks), dtype=np.float32)
        self._features = np.zeros((n_dates, self.n_stocks, self.n_features), dtype=np.float32)

        for i, code in enumerate(self.stock_codes):
            sub = df[df["code"] == code] if "code" in df.columns else df
            sub = sub.sort_values("date" if "date" in sub.columns else sub.index)

            for t, date in enumerate(self._dates):
                row = sub[sub["date"] == date] if "date" in sub.columns else (
                    sub.loc[[date]] if date in sub.index else None
                )
                if row is not None and not row.empty:
                    row = row.iloc[0]
                    self._prices[t, i] = float(row.get("close", 0))
                    for j, col in enumerate(self.feature_cols):
                        val = row.get(col, 0)
                        self._features[t, i, j] = float(val) if pd.notna(val) else 0.0

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """重置环境到起点"""
        super().reset(seed=seed)
        self.current_step = self.window
        self.portfolio_value = 1.0
        self.positions = np.zeros(self.n_stocks, dtype=np.float32)
        self.cash = 1.0
        self._returns_history = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步交易

        Args:
            action: (n_stocks, 2) 动作矩阵
                action[:, 0]: 方向 (-1卖出, 0持有, 1买入)
                action[:, 1]: 仓位比例 (0~1)

        Returns:
            (next_obs, reward, terminated, truncated, info)
        """
        step = self.current_step

        if step >= len(self._dates) - 1:
            return self._get_obs(), 0.0, True, False, {"early_stop": True}

        # 解包动作
        directions = np.clip(action[:, 0], -1, 1)  # -1 ~ 1
        position_ratios = np.clip(action[:, 1], 0, 1)  # 0 ~ 1

        # 当前价格
        current_prices = self._prices[step]
        next_prices = self._prices[step + 1]

        # 执行交易
        old_value = self.cash + np.sum(self.positions * current_prices)

        for i in range(self.n_stocks):
            if current_prices[i] <= 0 or next_prices[i] <= 0:
                continue

            target_ratio = (directions[i] + 1) / 2  # 0~1, 越高越看多
            target_pos = target_ratio * position_ratios[i] * self.max_position

            current_value = self.cash + np.sum(self.positions * current_prices)
            current_ratio = (self.positions[i] * current_prices[i]) / (current_value + 1e-8)
            diff_ratio = target_pos - current_ratio
            diff_value = diff_ratio * current_value

            if abs(diff_value) < current_prices[i] * 100:  # 最小交易单位
                continue

            if diff_value > 0:  # 买入
                # 考虑交易成本
                buy_value = min(diff_value, self.cash)
                buy_volume = buy_value / current_prices[i]
                cost = buy_value * self.transaction_cost
                if buy_volume > 0 and buy_value + cost <= self.cash:
                    self.positions[i] += buy_volume
                    self.cash -= buy_value + cost

            elif diff_value < 0:  # 卖出
                sell_volume = min(-diff_value / current_prices[i], self.positions[i])
                if sell_volume > 0:
                    sell_value = sell_volume * current_prices[i]
                    cost = sell_value * self.transaction_cost
                    self.positions[i] -= sell_volume
                    self.cash += sell_value - cost

        # 更新市值
        new_value = self.cash + np.sum(self.positions * next_prices)

        # 计算收益
        portfolio_return = (new_value - old_value) / (old_value + 1e-8)
        self._returns_history.append(portfolio_return)
        self.portfolio_value = new_value

        # 计算奖励
        reward = self._calculate_reward(directions)

        self.current_step += 1

        # 终止条件
        terminated = self.current_step >= self.max_steps
        truncated = self.portfolio_value < 0.3  # 亏损70%提前终止

        info = {
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "step": self.current_step,
            "return": float(portfolio_return),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """构建观测状态：过去 window 天的多股特征矩阵"""
        step = self.current_step
        start = max(0, step - self.window + 1)
        end = min(step + 1, len(self._prices))

        obs = np.zeros((self.window, self.n_stocks * self.n_features), dtype=np.float32)
        for t in range(start, end):
            idx = t - start
            # 展平: 将所有股票的特征拼接
            flat_features = self._features[t].flatten()
            obs[idx] = flat_features

        # 归一化
        obs_mean = obs.mean(axis=0, keepdims=True)
        obs_std = obs.std(axis=0, keepdims=True) + 1e-8
        obs = (obs - obs_mean) / obs_std

        return obs

    def _calculate_reward(self, directions: np.ndarray) -> float:
        """计算奖励：组合收益率 + 风险调整 + 交易惩罚

        综合奖励 = 收益率 * (1 - λ * 波动率) - 交易成本惩罚
        """
        if not self._returns_history:
            return 0.0

        # 近期收益率（最近一步）
        recent_returns = self._returns_history[-min(5, len(self._returns_history)):]
        mean_return = np.mean(recent_returns)

        # 波动率惩罚（用近期标准差）
        if len(recent_returns) > 1:
            vol = np.std(recent_returns) + 1e-8
            vol_penalty = 0.5 * vol
        else:
            vol_penalty = 0.0

        # 交易活跃度惩罚（鼓励稳定持仓）
        trade_activity = np.mean(np.abs(directions))
        trade_penalty = 0.01 * trade_activity

        # 最终奖励
        reward = mean_return - vol_penalty - trade_penalty

        return float(np.clip(reward, -1, 1))

    def render(self, mode="human"):
        """渲染（打印当前状态）"""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Portfolio: {self.portfolio_value:.4f}")
            print(f"Cash: {self.cash:.4f}")
            print(f"Positions: {np.sum(self.positions > 0)} stocks held")
