"""评估指标与跟踪。"""

from collections import defaultdict
from typing import Optional

import numpy as np


class MetricsTracker:
    """指标跟踪器。"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.all_metrics = defaultdict(list)

    def log(self, key: str, value: float, step: Optional[int] = None):
        """记录一个指标值。"""
        self.metrics[key].append(value)
        self.all_metrics[key].append(value)
        if len(self.metrics[key]) > self.window_size:
            self.metrics[key] = self.metrics[key][-self.window_size :]

    def get(self, key: str, window: bool = True) -> dict:
        """获取指标统计。"""
        data = self.metrics[key] if window else self.all_metrics[key]
        if not data:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "count": len(data),
            "last": float(data[-1]),
        }

    def get_all(self, window: bool = True) -> dict:
        """获取所有指标统计。"""
        return {key: self.get(key, window) for key in self.metrics}

    def reset(self):
        """重置所有指标。"""
        self.metrics.clear()
        self.all_metrics.clear()


def compute_metrics(predictions: list, references: list, task_type: str = "general") -> dict:
    """计算评估指标。"""
    assert len(predictions) == len(references), "预测和参考数量必须一致"

    n = len(predictions)
    if n == 0:
        return {"accuracy": 0.0, "count": 0}

    correct = sum(
        1 for pred, ref in zip(predictions, references) if _is_correct(pred, ref, task_type)
    )
    accuracy = correct / n

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
    }

    if isinstance(predictions[0], list):
        for k in [1, 5, 10]:
            pass_at_k = _compute_pass_at_k(predictions, references, k, task_type)
            metrics[f"pass@{k}"] = pass_at_k

    return metrics


def _is_correct(prediction, reference, task_type: str) -> bool:
    """判断单个预测是否正确。"""
    if task_type == "code":
        return prediction.get("passed", False) if isinstance(prediction, dict) else False
    elif task_type == "math":
        pred_ans = str(prediction).strip() if prediction else ""
        ref_ans = str(reference).strip() if reference else ""
        return pred_ans == ref_ans
    else:
        return str(prediction).strip() == str(reference).strip()


def _compute_pass_at_k(
    predictions: list[list], references: list, k: int, task_type: str
) -> float:
    """计算 Pass@k 指标。"""
    import math

    total = 0
    for preds, ref in zip(predictions, references):
        n = len(preds)
        c = sum(1 for pred in preds if _is_correct(pred, ref, task_type))
        if n - c < k:
            total += 1.0
        else:
            total += 1.0 - math.comb(n - c, k) / math.comb(n, k)
    return total / len(predictions) if predictions else 0.0
