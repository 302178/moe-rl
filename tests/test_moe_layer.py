"""MoE 层单元测试。"""

import torch
import pytest

from moe_rl.models.moe_layer import MoELayer, MoEConfig, Expert


class TestExpert:
    def test_expert_output_shape(self):
        expert = Expert(hidden_size=64, expert_hidden_size=128)
        x = torch.randn(4, 10, 64)
        out = expert(x)
        assert out.shape == (4, 10, 64)

    def test_expert_gradients(self):
        expert = Expert(hidden_size=64, expert_hidden_size=128)
        x = torch.randn(4, 10, 64, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert expert.w1.weight.grad is not None


class TestMoELayer:
    def setup_method(self):
        self.config = MoEConfig(
            num_experts=4,
            top_k=2,
            hidden_size=64,
            expert_hidden_size=128,
        )
        self.moe = MoELayer(self.config)

    def test_output_shape(self):
        x = torch.randn(2, 8, 64)
        out, loss = self.moe(x)
        assert out.shape == (2, 8, 64)

    def test_load_balancing_loss_in_training(self):
        self.moe.train()
        x = torch.randn(2, 8, 64)
        out, loss = self.moe(x)
        assert loss is not None
        assert loss.item() >= 0

    def test_no_load_balancing_loss_in_eval(self):
        self.moe.eval()
        x = torch.randn(2, 8, 64)
        out, loss = self.moe(x)
        assert loss is None

    def test_gradient_flow(self):
        self.moe.train()
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, loss = self.moe(x)
        total_loss = out.sum() + (loss if loss is not None else 0)
        total_loss.backward()
        assert x.grad is not None
        has_grad = any(p.grad is not None for p in self.moe.parameters())
        assert has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
