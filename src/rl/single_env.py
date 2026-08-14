"""rl/single_env.py — 单股交易环境

Gymnasium 环境，用于单只股票的 RL 买卖训练。

特性:
- Discrete(3) 动作: 持有(0) / 买入(1) / 卖出(2)
- 观察窗口: 60天 × 15个特征
- 持有时长约束: 最少3天, 最多30天
- P&L 盈亏驱动奖励
- A股规则: T+1, 手续费0.025%, 印花税0.1%(卖), 涨跌停±10%
"""

from typing import Optional, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


# 默认特征列（个股技术面 + 大盘/资金/换手率外部数据）
DEFAULT_FEATURES = [
    # ---- 价量基础 (3) ----
    "close",           # 收盘价
    "volume",          # 成交量
    "price_change",    # 涨跌幅%

    # ---- 均线族 (3) ----
    "MA5",             # 5日均线
    "MA20",            # 20日均线
    "MA60",            # 60日均线

    # ---- MACD 族 (3) ----
    "MACD_DIF",        # MACD DIF
    "MACD_DEA",        # MACD DEA
    "MACD_HIST",       # MACD 柱

    # ---- 超买超卖 (3) ----
    "RSI",             # RSI（14日）
    "CCI",             # 商品通道指数
    "KDJ_K",           # KDJ K值

    # ---- 布林+波动 (2) ----
    "BOLL_POSITION",   # 布林带 %B 位置
    "ATR_PCT",         # ATR 波动率百分比

    # ---- 成交量 (1) ----
    "VOL_RATIO",       # 量比（当日量/20日均量）

    # ---- 大盘环境 (3) ----
    "index_return_5d",      # 大盘近5日涨跌%
    "stock_vs_index_20d",   # 个股相对大盘20日超额收益
    "index_trend_20d",      # 大盘20日均线偏离%

    # ---- 资金流向 (2) ----
    "main_flow_pct",        # 主力净流入/成交额%
    "main_flow_5d",         # 近5日主力累计净流入/成交额%

    # ---- 换手率 (1) ----
    "turnover_rate",        # 换手率%
]


class SingleStockEnv(gym.Env):
    """单股 RL 交易环境

    Observation:
        Box(shape=(60, 15)) — 过去60天的15个特征

    Action:
        Discrete(3) — 0=持有, 1=买入, 2=卖出

    Reward:
        组合价值变化率 — 交易成本 — 违规惩罚
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        code: str = "",
        window: int = 60,
        feature_cols: list = None,
        commission: float = 0.00025,     # 佣金 0.025%
        stamp_tax: float = 0.001,        # 印花税 0.1% (仅卖出)
        min_hold_days: int = 3,
        max_hold_days: int = 30,
        initial_cash: float = 100_000.0,
        limit_pct: float = 0.10,         # 涨跌停 ±10%
    ):
        super().__init__()

        self.code = code
        self.window = window
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.initial_cash = initial_cash
        self.limit_pct = limit_pct

        # 特征列表
        self.feature_cols = feature_cols or DEFAULT_FEATURES

        # 提前补充 price_change（必须在特征过滤之前，否则会被当作缺失列丢弃）
        if "price_change" in self.feature_cols and "price_change" not in df.columns:
            close_vals = df["close"].astype(np.float32).values
            pc = np.zeros(len(df), dtype=np.float32)
            pc[1:] = (close_vals[1:] - close_vals[:-1]) / (close_vals[:-1] + 1e-8) * 100
            df = df.copy()
            df["price_change"] = pc

        # 只保留 df 中实际存在的列
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        self.n_features = len(self.feature_cols)

        if self.n_features < 5:
            raise ValueError(f"特征列不足: 需要至少5列，实际只有 {self.n_features} 列")

        # 预处理数据
        self._preprocess(df)

        # ---- 空间定义 ----
        # 观测: (window, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window, self.n_features),
            dtype=np.float32,
        )

        # 动作: Discrete(3) — 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # ---- 内部状态 ----
        self.current_step = window
        self.cash = initial_cash
        self.position = 0           # 持仓股数
        self.entry_price = 0.0      # 买入均价
        self.days_held = 0          # 连续持仓天数
        self.bought_today = False   # T+1: 当天买入不能卖
        self.portfolio_value = initial_cash
        self._prev_value = initial_cash
        self._peak_value = initial_cash   # 历史最高市值（用于回撤计算）
        self._max_drawdown = 0.0          # 最大回撤比例
        self._returns_history = []  # 收益率序列
        self._trades = []           # 交易记录
        self.max_steps = len(self._prices) - window - 1

    # ========================================================================
    # 数据预处理
    # ========================================================================

    def _preprocess(self, df: pd.DataFrame):
        """提取价格序列和特征矩阵"""
        # 按日期排序
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
            self._dates = pd.to_datetime(df["date"].tolist())
        else:
            self._dates = list(range(len(df)))

        n = len(df)

        # 价格序列（.copy() 防止 pandas 只读数组）
        if "close" in df.columns:
            self._prices = df["close"].astype(np.float32).values.copy()
        else:
            raise ValueError("数据缺少 'close' 列")

        # 成交量（用于判断涨跌停）
        if "volume" in df.columns:
            self._volumes = df["volume"].astype(np.float32).values.copy()
        else:
            self._volumes = np.ones(n, dtype=np.float32)

        # 自动补充 price_change（日涨跌幅%），如果数据中不存在
        if "price_change" not in df.columns:
            price_change = np.zeros(n, dtype=np.float32)
            price_change[1:] = (self._prices[1:] - self._prices[:-1]) / (self._prices[:-1] + 1e-8) * 100
            df = df.copy()
            df["price_change"] = price_change

        # 特征矩阵（.copy() 防止 pandas 只读数组导致 "assignment destination is read-only"）
        self._features = np.zeros((n, self.n_features), dtype=np.float32)
        for j, col in enumerate(self.feature_cols):
            if col in df.columns:
                vals = df[col].astype(np.float32).values.copy()
                # 填充 NaN
                mask = np.isnan(vals)
                if mask.any():
                    vals[mask] = 0.0
                self._features[:, j] = vals
            # 缺失的列保持为 0（已在 np.zeros 中预设）

        # 预计算每日涨跌幅（用于涨跌停检测）
        self._daily_returns = np.zeros(n, dtype=np.float32)
        self._daily_returns[1:] = (self._prices[1:] - self._prices[:-1]) / (self._prices[:-1] + 1e-8)

    # ========================================================================
    # Gym API
    # ========================================================================

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """重置环境到初始状态

        options:
            eval_mode: bool — True 时确定性从 window 起始（评估/推理用）
        """
        super().reset(seed=seed)

        eval_mode = options.get("eval_mode") if options else False

        # 最少需要 window + 1 行数据才能跑一步交易
        min_valid_step = min(self.window, len(self._prices) - 2)
        if eval_mode or self.max_steps <= 0:
            # 确定性起始：从 window 位置开始，最大化交易步数
            start = max(self.window, min_valid_step)
        else:
            # 随机选择起始点（训练时数据增强）
            lo = self.window
            hi = min(self.window + 120, len(self._prices) - 2)
            start = self.np_random.integers(lo, max(lo + 1, hi))

        self.current_step = start
        self.cash = self.initial_cash
        self.position = 0
        self.entry_price = 0.0
        self.days_held = 0
        self.bought_today = False
        self.portfolio_value = self.initial_cash
        self._prev_value = self.initial_cash
        self._peak_value = self.initial_cash
        self._max_drawdown = 0.0
        self._returns_history = []
        self._trades = []

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步交易

        Args:
            action: 0=持有, 1=买入, 2=卖出

        Returns:
            (next_obs, reward, terminated, truncated, info)
        """
        step = self.current_step
        current_price = float(self._prices[step])

        # ---- 检查是否到达数据末尾 ----
        if step >= len(self._prices) - 1:
            # 强制清算
            self._liquidate(current_price)
            return self._get_obs(), 0.0, True, False, {"reason": "data_end"}

        next_price = float(self._prices[step + 1])

        # ---- 检查涨跌停 ----
        limit_hit = self._check_limit(step, current_price)
        is_limit_up = limit_hit == "up"
        is_limit_down = limit_hit == "down"

        # ---- 执行动作 ----
        action_penalty = 0.0
        info = {"action": action, "position": self.position > 0}

        if action == 1:  # 买入
            if self.position > 0:
                # 已有持仓，忽略买入信号（单股单次持仓）
                action_penalty = -0.001
            elif is_limit_up:
                # 涨停买不到
                action_penalty = -0.001
            else:
                self._buy(current_price, step)

        elif action == 2:  # 卖出
            if self.position <= 0:
                # 无持仓可卖
                action_penalty = -0.001
            elif self.bought_today:
                # T+1 限制
                action_penalty = -0.002
            elif self.days_held < self.min_hold_days:
                # 持仓不足3天
                action_penalty = -0.003
            elif is_limit_down:
                # 跌停卖不掉
                action_penalty = -0.001
            else:
                self._sell(current_price, step)

        # ---- T+1 状态更新 ----
        self.bought_today = False  # 每天结束后重置

        # ---- 持仓天数 + 强制卖出检查 ----
        if self.position > 0:
            self.days_held += 1
            if self.days_held >= self.max_hold_days:
                # 超过30天，强制平仓
                self._sell(current_price, step, forced=True)
                action_penalty -= 0.005  # 超时惩罚

        # ---- 更新市值 ----
        self._prev_value = self.portfolio_value
        self.portfolio_value = self.cash + self.position * next_price

        # 更新峰值和最大回撤
        if self.portfolio_value > self._peak_value:
            self._peak_value = self.portfolio_value
        current_drawdown = (self._peak_value - self.portfolio_value) / (self._peak_value + 1e-10)
        if current_drawdown > self._max_drawdown:
            self._max_drawdown = current_drawdown

        # ---- 计算奖励（含回撤惩罚） ----
        reward = self._calc_reward(next_price, current_price, action_penalty, current_drawdown)

        # ---- 进入下一步 ----
        self.current_step += 1

        # 终止条件
        terminated = self.current_step >= len(self._prices) - 1
        truncated = self.portfolio_value < self.initial_cash * 0.5  # 亏50%截断

        if terminated and self.position > 0:
            self._liquidate(self._prices[-1])

        # info
        info.update({
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "position": int(self.position > 0),
            "days_held": self.days_held,
            "step": self.current_step,
            "pnl_pct": round((self.portfolio_value / self.initial_cash - 1) * 100, 2),
        })

        return self._get_obs(), float(reward), terminated, truncated, info

    # ========================================================================
    # 交易执行
    # ========================================================================

    def _buy(self, price: float, step: int):
        """全仓买入"""
        if self.cash <= price:
            return

        # 用全部现金买入（扣除手续费）
        affordable = self.cash / (1 + self.commission)
        shares = int(affordable / price)  # 按手计算: 100股/手
        shares = (shares // 100) * 100

        if shares <= 0:
            return

        cost = shares * price
        fee = cost * self.commission
        total_cost = cost + fee

        if total_cost > self.cash:
            return

        self.position = shares
        self.cash -= total_cost
        self.entry_price = price
        self.days_held = 0
        self.bought_today = True
        self._trades.append({"type": "buy", "price": float(price), "shares": shares,
                            "step": step, "fee": float(fee)})

    def _sell(self, price: float, step: int, forced: bool = False):
        """全仓卖出"""
        if self.position <= 0:
            return

        revenue = self.position * price
        commission_fee = revenue * self.commission
        stamp_fee = revenue * self.stamp_tax
        total_fee = commission_fee + stamp_fee

        self.cash += revenue - total_fee
        pnl = (price - self.entry_price) * self.position - total_fee
        pnl_pct = (price / self.entry_price - 1) * 100 if self.entry_price > 0 else 0

        self._trades.append({
            "type": "sell", "price": float(price),
            "shares": self.position, "step": step,
            "fee": float(total_fee), "pnl": float(pnl),
            "pnl_pct": round(float(pnl_pct), 2),
            "days_held": self.days_held, "forced": forced,
        })

        self.position = 0
        self.entry_price = 0.0
        self.days_held = 0

    def _liquidate(self, price: float):
        """强制清算（数据结束时）"""
        if self.position > 0:
            revenue = self.position * price
            commission_fee = revenue * self.commission
            stamp_fee = revenue * self.stamp_tax
            self.cash += revenue - commission_fee - stamp_fee
            self.position = 0

    # ========================================================================
    # 观测 & 奖励
    # ========================================================================

    def _get_obs(self) -> np.ndarray:
        """构建观测: 过去 window 天的特征矩阵，标准化处理"""
        step = self.current_step
        end = min(step + 1, len(self._features))
        start = max(0, end - self.window)

        obs = np.zeros((self.window, self.n_features), dtype=np.float32)
        actual_len = end - start
        obs[-actual_len:] = self._features[start:end]

        # Z-score 标准化
        mean = obs.mean(axis=0, keepdims=True)
        std = obs.std(axis=0, keepdims=True) + 1e-8
        obs = (obs - mean) / std

        # 位置编码: 最后一维加持仓标志
        # 在特征维度末尾附加持仓信息
        pos_flag = 1.0 if self.position > 0 else 0.0
        obs[:, -1] += pos_flag * 0.1  # 微弱信号，不破坏标准化

        return obs.astype(np.float32)

    def _calc_reward(self, next_price: float, current_price: float,
                     action_penalty: float = 0.0, drawdown: float = 0.0) -> float:
        """计算 P&L 驱动的奖励

        核心公式:
            reward = 组合价值变化率 — 波动率惩罚 — 回撤惩罚² — 仓位集中度惩罚 + 动作惩罚

        回撤惩罚权重随回撤深度非线性增长（drawdown²），阈值从 3% 开始。
        仓位集中度惩罚：100% 全仓时无惩罚，超过 100%（杠杆/借款）每 10% 额外 -0.001。
        """
        # 组合价值变化
        if self._prev_value > 0:
            pnl_return = (self.portfolio_value - self._prev_value) / self._prev_value
        else:
            pnl_return = 0.0

        # 记录收益率
        self._returns_history.append(pnl_return)

        # 近期波动率惩罚（降低方差）
        if len(self._returns_history) > 5:
            recent_returns = self._returns_history[-20:]
            if len(recent_returns) > 1:
                vol = np.std(recent_returns)
                vol_penalty = 0.1 * vol
            else:
                vol_penalty = 0.0
        else:
            vol_penalty = 0.0

        # 最大回撤惩罚 — 阈值降低到 3%，惩罚系数提高到 2.0
        # 回撤10% → 惩罚 0.02；回撤20% → 惩罚 0.08；回撤50% → 惩罚 0.5
        if drawdown > 0.03:
            dd_penalty = drawdown * drawdown * 2.0
        else:
            dd_penalty = 0.0

        # 仓位集中度惩罚 — 鼓励分散持仓，避免全仓一只股票
        total_value = self.cash + self.position * next_price
        if total_value > 0:
            position_ratio = (self.position * next_price) / total_value
            # >100% 则杠杆惩罚递增（正常全仓=1.0，无惩罚）
            if position_ratio > 1.0:
                concentration_penalty = (position_ratio - 1.0) * 0.01
            else:
                concentration_penalty = 0.0
        else:
            concentration_penalty = 0.0

        # 交易频率惩罚（持有奖励）
        hold_bonus = 0.0001 if self.position > 0 else 0.0

        # 最终奖励
        reward = pnl_return - vol_penalty - dd_penalty - concentration_penalty + hold_bonus + action_penalty

        # 收窄 clip 范围到 ±0.15（A股单日最大涨跌 ±10%，clip 应对极端情况即可）
        return float(np.clip(reward, -0.15, 0.15))

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _check_limit(self, step: int, price: float) -> Optional[str]:
        """检查涨跌停"""
        if step < 1:
            return None
        prev_price = float(self._prices[step - 1])
        if prev_price <= 0:
            return None

        change = (price - prev_price) / prev_price
        if change >= self.limit_pct - 0.001:
            return "up"
        elif change <= -self.limit_pct + 0.001:
            return "down"
        return None

    def render(self, mode="human"):
        """显示当前状态"""
        if mode == "human":
            pos_str = f"{self.position}股" if self.position > 0 else "空仓"
            pnl = (self.portfolio_value / self.initial_cash - 1) * 100
            print(
                f"Step:{self.current_step:4d} | "
                f"资产:¥{self.portfolio_value:,.0f} | "
                f"收益:{pnl:+.2f}% | "
                f"现金:¥{self.cash:,.0f} | "
                f"持仓:{pos_str} | "
                f"持天:{self.days_held}"
            )

    def get_trade_summary(self) -> dict:
        """获取交易汇总"""
        sells = [t for t in self._trades if t["type"] == "sell"]
        if not sells:
            return {"total_trades": 0, "win_rate": 0, "avg_return": 0,
                    "avg_hold_days": 0, "total_pnl": 0}

        wins = [t for t in sells if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in sells)
        avg_hold = sum(t["days_held"] for t in sells) / len(sells)

        return {
            "total_trades": len(sells),
            "win_rate": round(len(wins) / len(sells) * 100, 1),
            "avg_return_pct": round(sum(t["pnl_pct"] for t in sells) / len(sells), 2),
            "avg_hold_days": round(avg_hold, 1),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((self.portfolio_value / self.initial_cash - 1) * 100, 2),
            "max_drawdown_pct": round(self._max_drawdown * 100, 2),
        }
