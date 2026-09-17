"""训练器模块。"""

from .ppo_trainer import PPOTrainer, PPOConfig
from .grpo_trainer import GRPOTrainer, GRPOConfig

__all__ = ["PPOTrainer", "PPOConfig", "GRPOTrainer", "GRPOConfig"]
