"""Mixture of Experts 层实现。"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    """MoE 层配置。"""

    num_experts: int = 8
    top_k: int = 2
    hidden_size: int = 4096
    expert_hidden_size: int = 14336
    capacity_factor: float = 1.25
    dropout: float = 0.0
    load_balancing_loss_coef: float = 0.01
    router_jitter_noise: float = 0.0


class Expert(nn.Module):
    """单个专家网络（FFN）。"""

    def __init__(self, hidden_size: int, expert_hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.w2 = nn.Linear(expert_hidden_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act_fn(self.w1(x)) * self.w3(x)))


class MoELayer(nn.Module):
    """Mixture of Experts 层。

    支持 top-k 路由、负载均衡损失、容量限制。
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                Expert(config.hidden_size, config.expert_hidden_size, config.dropout)
                for _ in range(config.num_experts)
            ]
        )
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播。"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        router_logits = self.gate(hidden_states_flat)

        if self.training and self.config.router_jitter_noise > 0:
            router_logits += torch.empty_like(router_logits).uniform_(
                -self.config.router_jitter_noise, self.config.router_jitter_noise
            )

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        load_balancing_loss = None
        if self.training:
            load_balancing_loss = self._compute_load_balancing_loss(
                routing_weights, top_k_indices
            )

        output = torch.zeros_like(hidden_states_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = top_k_indices == expert_idx
            if not expert_mask.any():
                continue

            token_indices, k_positions = expert_mask.nonzero(as_tuple=True)
            if len(token_indices) == 0:
                continue

            expert_input = hidden_states_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)

            weights = top_k_weights[token_indices, k_positions].unsqueeze(-1)
            output[token_indices] += expert_output * weights

        output = output.view(batch_size, seq_len, hidden_size)
        return output, load_balancing_loss

    def _compute_load_balancing_loss(
        self, routing_weights: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """计算负载均衡损失。"""
        num_tokens = routing_weights.shape[0]

        avg_probs = routing_weights.mean(dim=0)

        expert_counts = torch.zeros(self.num_experts, device=routing_weights.device)
        for k in range(self.top_k):
            expert_counts += torch.bincount(
                top_k_indices[:, k], minlength=self.num_experts
            ).float()
        expert_frac = expert_counts / (num_tokens * self.top_k)

        loss = self.num_experts * (avg_probs * expert_frac).sum()
        return self.config.load_balancing_loss_coef * loss
