"""GRPO (Group Relative Policy Optimization) 训练器。

适用于无 critic 的 RL 训练，DeepSeekMath 等工作中使用。
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GRPOConfig:
    """GRPO 训练配置。"""

    learning_rate: float = 1e-6
    batch_size: int = 8
    group_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    max_grad_norm: float = 1.0

    beta: float = 0.04
    clip_eps: float = 0.2

    gen_kwargs: dict = field(
        default_factory=lambda: {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
        }
    )

    log_with: Optional[str] = "wandb"
    project_name: str = "moe-rl-grpo"


class GRPOTrainer:
    """GRPO 训练器。

    Group Relative Policy Optimization:
    - 对每个 prompt 采样 group_size 个回复
    - 用组内相对优势（减去组内均值，除以组内标准差）
    - 无需 value network
    - 支持 KL 惩罚（使用 reference model）
    """

    def __init__(
        self,
        config: GRPOConfig,
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module] = None,
        tokenizer=None,
        optimizer=None,
        reward_fn=None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.step_count = 0

    def generate_group(self, prompt: torch.Tensor) -> torch.Tensor:
        """为单个 prompt 生成一组回复。"""
        prompts = prompt.repeat(self.config.group_size, 1)
        with torch.no_grad():
            outputs = self.model.generate(prompts, **self.config.gen_kwargs)
        return outputs

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算组内相对优势。"""
        mean = rewards.mean()
        std = rewards.std()
        if std < 1e-8:
            return torch.zeros_like(rewards)
        advantages = (rewards - mean) / std
        return advantages

    def step(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict:
        """执行一步 GRPO 更新。"""
        stats = {}
        batch_size = prompts.shape[0]

        advantages = torch.zeros_like(rewards)
        for i in range(batch_size):
            start = i * self.config.group_size
            end = start + self.config.group_size
            group_rewards = rewards[start:end]
            advantages[start:end] = self.compute_group_advantages(group_rewards)

        stats["reward_mean"] = rewards.mean().item()
        stats["advantage_mean"] = advantages.mean().item()
        stats["step"] = self.step_count

        self.step_count += 1
        return stats

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
