"""rl/train.py — 单股 RL 训练管道

支持:
- 全量训练: 2年历史数据
- 增量更新: 15天内模型微调
- 自动过期: 60天未更新自动删除

生命周期元数据: models/{code}_meta.json
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from .single_env import SingleStockEnv, DEFAULT_FEATURES

logger = logging.getLogger(__name__)

# 可从 config 覆盖的环境参数（SingleStockEnv.__init__）
_ENV_PARAMS = ("window", "min_hold_days", "max_hold_days", "commission", "stamp_tax", "limit_pct")


def _extract_env_kwargs(config: dict) -> dict:
    """从训练 config 提取环境参数，供 SingleStockEnv 使用"""
    return {k: config[k] for k in _ENV_PARAMS if config.get(k) is not None}

# 默认配置
DEFAULT_CONFIG = {
    "model_dir": "models",
    "update_interval_days": 15,
    "delete_stale_days": 60,
    "train_data_years": 2,
    "min_data_rows": 120,
    "timesteps_full": 200_000,
    "timesteps_incremental": 10_000,
    "ppo_kwargs": {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "batch_size": 64,
        "n_steps": 2048,
    },
}


# ============================================================================
# 主训练入口（生命周期感知）
# ============================================================================

def train_single_stock(
    df: pd.DataFrame,
    code: str,
    timesteps: int = None,
    force_full: bool = False,
    model_dir: str = "models",
    config: dict = None,
    extra_callbacks: list = None,
    cancel_event=None,
) -> Dict:
    """单股生命周期感知训练

    决策树:
        - 首次训练 → 全量训练
        - < 15天 + 模型存在 → 增量更新
        - 15~60天 → 全量重训
        - > 60天 → 删除旧模型 → 全量训练

    Args:
        df: 历史数据 (需含 date, close 及技术指标)
        code: 股票代码
        timesteps: 训练步数 (None=使用默认值)
        force_full: 强制全量训练
        model_dir: 模型保存目录
        extra_callbacks: (可选) 额外 SB3 回调列表，用于 GUI 训练进度/取消
        cancel_event: (可选) 可调用对象，返回 True 表示用户已取消

    Returns:
        {"status": "ok"|"updated"|"already_fresh", "mode": "full"|"incremental", ...}
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    model_dir = model_dir or cfg["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{code}_ppo")
    meta_path = os.path.join(model_dir, f"{code}_meta.json")

    # ---- 检查元数据 ----
    if os.path.exists(meta_path) and not force_full:
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            last_train = datetime.fromisoformat(meta.get("last_train_date", "2000-01-01"))
            days_since = (datetime.now() - last_train).days

            # > 60天 → 删除旧模型
            if days_since > cfg["delete_stale_days"]:
                logger.info(f"{code}: 模型过期 {days_since} 天，删除重建")
                _delete_model_assets(model_path, meta_path)
                return _full_train(df, code, model_path, meta_path,
                                   timesteps or cfg["timesteps_full"], cfg,
                                   extra_callbacks=extra_callbacks,
                                   cancel_event=cancel_event)

            # < 15天 → 增量更新
            if days_since <= cfg["update_interval_days"]:
                if os.path.exists(f"{model_path}.zip"):
                    logger.info(f"{code}: 距上次训练 {days_since} 天，增量更新")
                    return _incremental_train(df, code, model_path, meta_path,
                                              timesteps or cfg["timesteps_incremental"], cfg,
                                              extra_callbacks=extra_callbacks,
                                              cancel_event=cancel_event)
                else:
                    logger.warning(f"{code}: 元数据在但模型文件缺失，全量重训")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"{code}: 元数据损坏 ({e})，全量重训")

    # ---- 全量训练 ----
    return _full_train(df, code, model_path, meta_path,
                       timesteps or cfg["timesteps_full"], cfg,
                       extra_callbacks=extra_callbacks,
                       cancel_event=cancel_event)


# ============================================================================
# 全量训练
# ============================================================================

def _full_train(
    df: pd.DataFrame,
    code: str,
    model_path: str,
    meta_path: str,
    timesteps: int,
    config: dict,
    extra_callbacks: list = None,
    cancel_event=None,
) -> Dict:
    """全量训练：时序划分训练/验证集 + 早停 + 验证集评估"""
    # 数据校验
    if df.empty:
        return {"status": "error", "code": code, "message": "数据为空"}
    min_rows = config.get("min_data_rows", 120)
    if len(df) < min_rows:
        return {"status": "error", "code": code,
                "message": f"数据不足 (需≥{min_rows}行, 实际{len(df)}行)"}

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
        from stable_baselines3.common.monitor import Monitor

        # ---- 时序划分：训练集=前80%，验证集=后20%（至少150天保证有意义的评估） ----
        # 验证集至少需要 window(60) + 交易周期(90) = 150 天，否则评估结果无意义
        val_size = max(150, min(int(len(df) * 0.25), 250))
        if val_size >= len(df) - 60:
            val_size = max(60, len(df) // 3)  # 数据不足时放宽
        train_df = df.iloc[:-val_size].reset_index(drop=True)
        val_df = df.iloc[-val_size:].reset_index(drop=True)

        if len(df) < 250:
            logger.warning(f"{code}: 数据较少 ({len(df)}行)，验证集可能不足以支持完整评估")

        logger.info(f"{code}: 训练集 {len(train_df)} 行, 验证集 {len(val_df)} 行")

        # 创建环境（val_env 用 Monitor 包装，EvalCallback 依赖它正确统计）
        env_kwargs = _extract_env_kwargs(config)
        train_env = SingleStockEnv(train_df, code=code, **env_kwargs)
        val_env = Monitor(SingleStockEnv(val_df, code=code, **env_kwargs))
        # PPO 超参深合并：config 只给部分键时保留默认值
        ppo_kwargs = {**DEFAULT_CONFIG["ppo_kwargs"], **(config.get("ppo_kwargs") or {})}

        # 早停回调：连续8次无改善即停止（放宽到8次，配合修复后的评估）
        # min_evals=15 确保至少跑 37.5% 的评估后才考虑早停
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=8,
            min_evals=15,
            verbose=0,
        )
        eval_freq = max(8000, timesteps // 25)  # 每4%步数评估一次
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=None,
            eval_freq=eval_freq,
            callback_after_eval=stop_callback,
            verbose=0,
            deterministic=True,
        )

        # 创建模型
        model = PPO("MlpPolicy", train_env, verbose=0, **ppo_kwargs)

        # 训练（含早停 + 可选进度回调）
        callbacks = [eval_callback] + list(extra_callbacks or [])
        logger.info(f"{code}: 全量训练 {timesteps:,} 步 (eval每{eval_freq}步)...")
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

        # 取消检查：取消后不保存模型
        if cancel_event is not None and callable(cancel_event) and cancel_event():
            logger.info(f"{code}: 训练被用户取消，不保存模型")
            return {"status": "cancelled", "code": code,
                    "message": "训练已取消（未保存模型）"}

        # 训练步数（可能因早停而提前结束）
        actual_steps = min(timesteps, model.num_timesteps)

        # ---- 在验证集上评估 ----
        eval_metrics = evaluate_model(model, val_df, code)

        # 保存模型
        model.save(model_path)
        logger.info(f"{code}: 模型已保存 → {model_path}.zip (实际训练{actual_steps:,}步)")

        # 保存元数据
        meta = {
            "code": code,
            "last_train_date": datetime.now().isoformat(),
            "total_timesteps": actual_steps,
            "target_timesteps": timesteps,
            "mode": "full",
            "data_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "feature_count": train_env.n_features,
            "update_count": 0,
            "eval": eval_metrics,
            "early_stopped": actual_steps < timesteps,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {
            "status": "ok",
            "code": code,
            "mode": "full_train",
            "timesteps": actual_steps,
            "target_timesteps": timesteps,
            "data_rows": len(df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "data_range": _get_data_range(df),
            "train_range": _get_data_range(train_df),
            "val_range": _get_data_range(val_df),
            "early_stopped": actual_steps < timesteps,
            "eval_metrics": eval_metrics,
            "model_path": f"{model_path}.zip",
        }

    except ImportError as e:
        return {"status": "error", "code": code,
                "message": f"依赖缺失: {e}\n请安装: pip install stable-baselines3"}
    except Exception as e:
        logger.exception(f"{code}: 训练失败")
        return {"status": "error", "code": code, "message": str(e)}


# ============================================================================
# 增量训练
# ============================================================================

def _incremental_train(
    df: pd.DataFrame,
    code: str,
    model_path: str,
    meta_path: str,
    timesteps: int,
    config: dict,
    extra_callbacks: list = None,
    cancel_event=None,
) -> Dict:
    """增量微调：加载已有模型，用新数据继续训练"""
    try:
        from stable_baselines3 import PPO

        # 加载已有模型
        logger.info(f"{code}: 加载已有模型 {model_path}")
        model = PPO.load(model_path)

        # 用新数据创建环境
        env = SingleStockEnv(df, code=code, **_extract_env_kwargs(config))
        model.set_env(env)

        # 微调（+ 可选进度回调）
        callbacks = list(extra_callbacks or [])
        logger.info(f"{code}: 增量训练 {timesteps:,} 步...")
        model.learn(total_timesteps=timesteps, callback=callbacks or None,
                    progress_bar=False)

        # 取消检查：取消后不保存模型
        if cancel_event is not None and callable(cancel_event) and cancel_event():
            logger.info(f"{code}: 增量训练被用户取消，不保存模型")
            return {"status": "cancelled", "code": code,
                    "message": "训练已取消（未保存模型）"}

        # 保存
        model.save(model_path)

        # ---- 增量训练后也做评估 ----
        val_size = max(150, min(int(len(df) * 0.25), 250))
        if val_size >= len(df) - 60:
            val_size = max(60, len(df) // 3)
        val_df = df.iloc[-val_size:].reset_index(drop=True)
        eval_metrics = evaluate_model(model, val_df, code)

        # 更新元数据
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["last_train_date"] = datetime.now().isoformat()
        meta["update_count"] = meta.get("update_count", 0) + 1
        meta["mode"] = "incremental"
        meta["eval"] = eval_metrics
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {
            "status": "updated",
            "code": code,
            "mode": "incremental",
            "timesteps": timesteps,
            "update_count": meta["update_count"],
            "previous_train_date": meta.get("last_train_date", ""),
            "model_path": f"{model_path}.zip",
            "eval_metrics": eval_metrics,
        }

    except Exception as e:
        logger.warning(f"{code}: 增量更新失败 ({e})，回退到全量训练")
        return _full_train(df, code, model_path, meta_path,
                          config.get("timesteps_full", DEFAULT_CONFIG["timesteps_full"]), config,
                          extra_callbacks=extra_callbacks,
                          cancel_event=cancel_event)


# ============================================================================
# 模型评估
# ============================================================================

def evaluate_model(model, df: pd.DataFrame, code: str = "",
                   seed: int = 42) -> Dict:
    """确定性 walk-forward 评估：从验证集起点跑到终点

    解决的问题：
    - 旧版直接 env.reset() 随机起始位置，验证集120行时常被分到只剩1-3步
    - 3步内受 min_hold_days=3 约束无法完成一次完整买卖 → 永远 0 trades
    - 现在用 eval_mode=True 确定性从 window 位置起始，跑完整验证集

    时间序列评估不适合多 episode 切分（每段太短无法交易），
    应使用单次完整 walk-forward 评估。

    Args:
        model: PPO 模型
        df: 评估数据（如验证集）
        code: 股票代码
        seed: 随机种子（确定性）

    Returns:
        {"total_return_pct": ..., "sharpe": ..., "trades": ..., "win_rate": ...}
    """
    env = SingleStockEnv(df, code=code)
    # 确定性起始：eval_mode=True 确保从 window 位置开始，跑满全段
    obs, _ = env.reset(seed=seed, options={"eval_mode": True})

    done = False
    total_reward = 0.0
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        rewards.append(reward)
        done = terminated or truncated

    trade_summary = env.get_trade_summary()

    # 计算夏普比率（年化近似）
    if len(rewards) > 1:
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        sharpe = (mean_r / std_r) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "total_return_pct": trade_summary.get("total_return_pct", 0),
        "sharpe": round(sharpe, 2),
        "trades": trade_summary.get("total_trades", 0),
        "win_rate": trade_summary.get("win_rate", 0),
        "avg_hold_days": trade_summary.get("avg_hold_days", 0),
        "total_pnl": trade_summary.get("total_pnl", 0),
        "max_drawdown_pct": trade_summary.get("max_drawdown_pct", 0),
        "steps": len(rewards),
    }


# ============================================================================
# 模型加载（用于推理）
# ============================================================================

def load_single_model(code: str, model_dir: str = "models") -> Optional[any]:
    """加载单股 RL 模型

    Returns:
        PPO 模型实例，或 None (未训练/文件不存在)
    """
    model_path = os.path.join(model_dir, f"{code}_ppo")
    if not os.path.exists(f"{model_path}.zip"):
        return None

    try:
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    except Exception as e:
        logger.warning(f"加载 {code} 模型失败: {e}")
        return None


def check_model_freshness(code: str, model_dir: str = "models") -> Dict:
    """检查模型新鲜度

    Returns:
        {"exists": bool, "days_since_train": int|None, "is_fresh": bool, "is_stale": bool}
    """
    meta_path = os.path.join(model_dir, f"{code}_meta.json")

    if not os.path.exists(meta_path):
        return {"exists": False, "days_since_train": None,
                "is_fresh": False, "is_stale": False}

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        last_train = datetime.fromisoformat(meta.get("last_train_date", "2000-01-01"))
        days_since = (datetime.now() - last_train).days

        model_exists = os.path.exists(os.path.join(model_dir, f"{code}_ppo.zip"))

        return {
            "exists": model_exists,
            "days_since_train": days_since,
            "is_fresh": days_since <= DEFAULT_CONFIG["update_interval_days"],
            "is_stale": days_since > DEFAULT_CONFIG["delete_stale_days"],
            "last_train_date": meta.get("last_train_date"),
            "eval_return": meta.get("eval", {}).get("total_return_pct"),
        }
    except Exception:
        return {"exists": False, "days_since_train": None,
                "is_fresh": False, "is_stale": False}


# ============================================================================
# 工具函数
# ============================================================================

def _get_data_range(df: pd.DataFrame) -> str:
    """获取数据日期范围"""
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        return f"{dates.min().date()} ~ {dates.max().date()}"
    return f"{len(df)} 行"


def _delete_model_assets(model_path: str, meta_path: str):
    """删除模型及元数据"""
    for p in [f"{model_path}.zip", meta_path]:
        if os.path.exists(p):
            os.remove(p)
            logger.info(f"已删除: {p}")


# ============================================================================
# 遗留兼容：多股训练（保持 API 不变但不再推荐使用）
# ============================================================================

def train(
    df: pd.DataFrame,
    stock_codes: list,
    feature_cols: list,
    total_timesteps: int = 100000,
    model_path: str = "models/ppo_stock",
    config: Optional[dict] = None,
) -> Dict:
    """多股联合训练（遗留，建议使用 train_single_stock）"""
    from .env import MultiStockTradingEnv
    from .model import create_model, save_model

    if df.empty:
        return {"status": "error", "message": "训练数据为空"}

    cfg = config or {}

    train_env = MultiStockTradingEnv(
        df=df, stock_codes=stock_codes, feature_cols=feature_cols,
        window=cfg.get("window", 60),
        transaction_cost=cfg.get("transaction_cost", 0.0003),
        max_position=cfg.get("max_position", 0.2),
    )

    model = create_model(
        env=train_env, algorithm=cfg.get("algorithm", "PPO"),
        learning_rate=cfg.get("learning_rate", 0.0003),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
    )

    logger.info(f"开始训练 [{cfg.get('algorithm', 'PPO')}] {total_timesteps} 步...")
    model.learn(total_timesteps=total_timesteps)
    save_model(model, model_path)

    return {
        "status": "ok", "model_path": model_path,
        "algorithm": cfg.get("algorithm", "PPO"),
        "total_timesteps": total_timesteps,
        "stock_codes": stock_codes,
        "feature_count": len(feature_cols),
    }
