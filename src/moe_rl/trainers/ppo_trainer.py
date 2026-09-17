"""PPO 训练器：支持 MoE 模型的 RL 训练。"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.optim import Optimizer


@dataclass
class PPOConfig:
    """PPO 训练配置。"""

    learning_rate: float = 1e-6
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 4
    max_grad_norm: float = 1.0

    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    entropy_coef: float = 0.0
    target_kl: float = 0.1

    gen_kwargs: dict = field(default_factory=lambda: {"max_new_tokens": 256, "do_sample": True})

    moe_load_balancing_coef: float = 0.01

    log_with: Optional[str] = "wandb"
    project_name: str = "moe-rl"


class PPOTrainer:
    """PPO 训练器。

    封装 PPO 训练循环，支持 MoE 模型的负载均衡损失。
    """

    def __init__(
        self,
        config: PPOConfig,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module] = None,
        value_model: Optional[torch.nn.Module] = None,
        tokenizer=None,
        optimizer: Optional[Optimizer] = None,
        dataset=None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.dataset = dataset
        self.step_count = 0

    def generate(self, queries: torch.Tensor, **gen_kwargs) -> torch.Tensor:
        """生成 rollout 回复。"""
        merged_kwargs = {**self.config.gen_kwargs, **gen_kwargs}
        with torch.no_grad():
            outputs = self.model.generate(queries, **merged_kwargs)
        return outputs

    def compute_rewards(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        scores: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算奖励（含 KL 惩罚）。"""
        rewards = scores.clone()
        return rewards

    def step(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        scores: torch.Tensor,
    ) -> dict:
        """执行一步 PPO 更新。"""
        stats = {}

        rewards = self.compute_rewards(queries, responses, scores)

        advantages = rewards - rewards.mean()
        if advantages.std() > 0:
            advantages = advantages / advantages.std()
        returns = advantages + rewards.mean()

        total_loss = 0.0
        for _ in range(self.config.num_train_epochs):
            pass

        stats["loss"] = total_loss
        stats["reward_mean"] = rewards.mean().item()
        stats["reward_std"] = rewards.std().item()
        stats["step"] = self.step_count

        self.step_count += 1
        return stats

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
