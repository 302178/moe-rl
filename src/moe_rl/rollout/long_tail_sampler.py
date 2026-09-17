"""长尾采样器：针对 RL rollout 中的 hard examples 进行专项采样。"""

from collections import deque
from typing import Optional

import numpy as np
import torch


class LongTailSampler:
    """长尾样本采样器。

    核心思路：
    1. 维护一个 hard example 池（低奖励、高损失、罕见模式）
    2. 按一定比例从 hard pool 和普通数据中采样
    3. 支持难度估计和动态调整

    适用于解决 RL rollout 中的长尾问题。
    """

    def __init__(
        self,
        hard_example_ratio: float = 0.3,
        pool_size: int = 10000,
        difficulty_threshold: float = 0.3,
        difficulty_metric: str = "reward",
    ):
        self.hard_example_ratio = hard_example_ratio
        self.pool_size = pool_size
        self.difficulty_threshold = difficulty_threshold
        self.difficulty_metric = difficulty_metric

        self.hard_pool = deque(maxlen=pool_size)
        self.hard_scores = deque(maxlen=pool_size)

        self.total_sampled = 0
        self.hard_sampled = 0

    def estimate_difficulty(
        self,
        example: dict,
        reward: Optional[float] = None,
        loss: Optional[float] = None,
    ) -> float:
        """估计样本难度。难度分数越高表示越难。"""
        if self.difficulty_metric == "reward" and reward is not None:
            return 1.0 - max(0.0, min(1.0, reward))
        elif self.difficulty_metric == "loss" and loss is not None:
            return min(1.0, loss / (loss + 1.0))
        elif self.difficulty_metric == "combined":
            score = 0.0
            if reward is not None:
                score += 0.5 * (1.0 - max(0.0, min(1.0, reward)))
            if loss is not None:
                score += 0.5 * min(1.0, loss / (loss + 1.0))
            return score
        else:
            return 0.5

    def add_to_hard_pool(self, example: dict, difficulty_score: float):
        """将样本加入 hard pool。"""
        if difficulty_score >= self.difficulty_threshold:
            self.hard_pool.append(example)
            self.hard_scores.append(difficulty_score)

    def update_from_rollout(self, rollout_results: list[dict]):
        """从 rollout 结果中更新 hard pool。"""
        for result in rollout_results:
            reward = result.get("reward")
            loss = result.get("loss")
            difficulty = self.estimate_difficulty(result, reward, loss)
            self.add_to_hard_pool(result, difficulty)

    def sample(self, batch_size: int, regular_data: Optional[list] = None) -> list[dict]:
        """采样一个 batch，包含一定比例的 hard examples。"""
        num_hard = int(batch_size * self.hard_example_ratio)
        num_regular = batch_size - num_hard

        batch = []

        if len(self.hard_pool) > 0 and num_hard > 0:
            scores = np.array(self.hard_scores)
            probs = scores / scores.sum() if scores.sum() > 0 else None
            hard_indices = np.random.choice(
                len(self.hard_pool),
                size=min(num_hard, len(self.hard_pool)),
                replace=False,
                p=probs,
            )
            for idx in hard_indices:
                batch.append(self.hard_pool[idx])
                self.hard_sampled += 1

        if regular_data is not None and num_regular > 0:
            regular_indices = np.random.choice(
                len(regular_data), size=min(num_regular, len(regular_data)), replace=False
            )
            for idx in regular_indices:
                batch.append(regular_data[idx])

        while len(batch) < batch_size and len(self.hard_pool) > 0:
            idx = np.random.randint(len(self.hard_pool))
            batch.append(self.hard_pool[idx])

        self.total_sampled += len(batch)
        np.random.shuffle(batch)
        return batch

    def get_stats(self) -> dict:
        """获取采样统计。"""
        return {
            "hard_pool_size": len(self.hard_pool),
            "total_sampled": self.total_sampled,
            "hard_sampled": self.hard_sampled,
            "hard_ratio": (
                self.hard_sampled / self.total_sampled if self.total_sampled > 0 else 0
            ),
            "avg_difficulty": (
                float(np.mean(self.hard_scores)) if len(self.hard_scores) > 0 else 0
            ),
        }

    def reset(self):
        """重置采样器。"""
        self.hard_pool.clear()
        self.hard_scores.clear()
        self.total_sampled = 0
        self.hard_sampled = 0
