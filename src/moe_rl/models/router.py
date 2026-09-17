"""路由器模块：多种路由策略实现。"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(ABC, nn.Module):
    """路由器基类。"""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

    @abstractmethod
    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """路由。"""
        raise NotImplementedError


class TopKRouter(Router):
    """标准 Top-K 路由器。"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
    ):
        super().__init__(hidden_size, num_experts)
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        router_logits = self.gate(hidden_states)

        if self.training and self.jitter_noise > 0:
            router_logits += torch.empty_like(router_logits).uniform_(
                -self.jitter_noise, self.jitter_noise
            )

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        return top_k_weights, top_k_indices, None
