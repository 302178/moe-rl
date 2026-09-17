"""长尾采样器单元测试。"""

import pytest

from moe_rl.rollout.long_tail_sampler import LongTailSampler


class TestLongTailSampler:
    def setup_method(self):
        self.sampler = LongTailSampler(
            hard_example_ratio=0.5,
            pool_size=100,
            difficulty_threshold=0.3,
        )

    def test_initial_state(self):
        stats = self.sampler.get_stats()
        assert stats["hard_pool_size"] == 0
        assert stats["total_sampled"] == 0
        assert stats["hard_sampled"] == 0

    def test_estimate_difficulty_by_reward(self):
        self.sampler.difficulty_metric = "reward"
        d1 = self.sampler.estimate_difficulty({}, reward=0.0)
        d2 = self.sampler.estimate_difficulty({}, reward=1.0)
        assert d1 > d2
        assert d1 == 1.0
        assert d2 == 0.0

    def test_add_to_hard_pool(self):
        example = {"prompt": "test", "id": 1}
        self.sampler.add_to_hard_pool(example, difficulty_score=0.8)
        assert len(self.sampler.hard_pool) == 1
        assert self.sampler.hard_pool[0]["id"] == 1

    def test_not_add_below_threshold(self):
        example = {"prompt": "test", "id": 1}
        self.sampler.add_to_hard_pool(example, difficulty_score=0.1)
        assert len(self.sampler.hard_pool) == 0

    def test_pool_size_limit(self):
        sampler = LongTailSampler(pool_size=5)
        for i in range(10):
            sampler.add_to_hard_pool({"id": i}, difficulty_score=0.8)
        assert len(sampler.hard_pool) == 5
        assert sampler.hard_pool[0]["id"] == 5

    def test_sample_from_hard_pool(self):
        for i in range(10):
            self.sampler.add_to_hard_pool({"id": i}, difficulty_score=0.5 + i * 0.05)

        batch = self.sampler.sample(batch_size=4, regular_data=None)
        assert len(batch) == 4
        assert all("id" in item for item in batch)

    def test_update_from_rollout(self):
        rollout_results = [
            {"prompt": "hard1", "reward": 0.0, "loss": 5.0},
            {"prompt": "easy1", "reward": 1.0, "loss": 0.1},
            {"prompt": "hard2", "reward": 0.2, "loss": 3.0},
        ]
        self.sampler.difficulty_metric = "reward"
        self.sampler.update_from_rollout(rollout_results)
        assert len(self.sampler.hard_pool) >= 1

    def test_reset(self):
        self.sampler.add_to_hard_pool({"id": 1}, difficulty_score=0.8)
        self.sampler.total_sampled = 10
        self.sampler.hard_sampled = 5
        self.sampler.reset()
        stats = self.sampler.get_stats()
        assert stats["hard_pool_size"] == 0
        assert stats["total_sampled"] == 0
        assert stats["hard_sampled"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
