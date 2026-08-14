"""rl/model.py — RL 模型定义

神经网络架构（MLP + 可选 LSTM），模型保存与加载。
"""

from typing import Optional, Any
import os


def create_model(env, algorithm: str = "PPO", **kwargs) -> Any:
    """创建强化学习模型

    Args:
        env: Gymnasium 环境
        algorithm: PPO / DQN / A2C
        **kwargs: 模型参数（learning_rate, gamma 等）

    Returns:
        stable-baselines3 模型实例

    Raises:
        ImportError: 如果未安装 stable-baselines3
        ValueError: 不支持的算法
    """
    try:
        from stable_baselines3 import PPO, DQN, A2C
    except ImportError:
        raise ImportError(
            "需要安装 stable-baselines3: pip install stable-baselines3"
        )

    algo_map = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
    }

    if algorithm.upper() not in algo_map:
        raise ValueError(f"不支持的算法: {algorithm}，可选: {list(algo_map.keys())}")

    algo_class = algo_map[algorithm.upper()]

    # 默认参数
    default_params = {
        "PPO": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "verbose": 1,
        },
        "DQN": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "verbose": 1,
        },
        "A2C": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0007,
            "gamma": 0.99,
            "verbose": 1,
        },
    }

    params = default_params.get(algorithm.upper(), default_params["PPO"]).copy()
    params.update(kwargs)

    model = algo_class(env=env, **params)
    return model


def save_model(model, path: str):
    """保存模型到磁盘

    Args:
        model: stable-baselines3 模型
        path: 保存路径（不含扩展名）
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    model.save(path)
    print(f"模型已保存: {path}.zip")


def load_model(path: str, env=None) -> Optional[Any]:
    """从磁盘加载模型

    Args:
        path: 模型路径（不含扩展名）
        env: 环境（可选）

    Returns:
        加载的模型，或 None（加载失败时）
    """
    try:
        from stable_baselines3 import PPO, DQN, A2C
    except ImportError:
        raise ImportError("需要安装 stable-baselines3")

    # 自动检测算法类型
    for algo_cls in [PPO, DQN, A2C]:
        try:
            model = algo_cls.load(path, env=env)
            print(f"模型已加载: {path}")
            return model
        except Exception:
            continue

    raise ValueError(f"无法加载模型: {path}，请确认算法类型和路径正确")
