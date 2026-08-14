"""data/preprocessor.py — 数据预处理

多股数据对齐、缺失值处理、归一化、训练/测试集划分。
"""

from typing import Tuple, Optional


def align_multi_stock(data_dict: dict) -> "pd.DataFrame":
    """将多只股票数据按日期对齐，拼接为宽表

    Args:
        data_dict: {code: DataFrame} 每只股票的日线数据

    Returns:
        对齐后的 MultiIndex DataFrame (date, code)
    """
    pass


def fill_missing(df: "pd.DataFrame", method: str = "ffill") -> "pd.DataFrame":
    """填充缺失值"""
    pass


def normalize(
    df: "pd.DataFrame", method: str = "zscore"
) -> Tuple["pd.DataFrame", dict]:
    """归一化处理

    Args:
        method: "zscore" / "minmax"

    Returns:
        (归一化后数据, 归一化参数)
    """
    pass


def train_test_split(
    df: "pd.DataFrame", train_ratio: float = 0.8
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """按时间顺序划分训练/测试集（不随机打乱）"""
    pass


def build_observation(
    df: "pd.DataFrame", window: int = 60
) -> "np.ndarray":
    """构建 RL 模型输入的观测窗口

    将过去 window 天的多股数据堆叠为状态矩阵。

    Returns:
        shape: (samples, window, n_stocks * n_features)
    """
    pass
