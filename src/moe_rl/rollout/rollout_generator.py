"""Rollout 生成器：管理 RL 训练中的采样过程。"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class RolloutConfig:
    """Rollout 配置。"""

    batch_size: int = 4
    num_return_sequences: int = 1
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    repetition_penalty: float = 1.0

    long_tail_focus: bool = False
    hard_example_ratio: float = 0.3
    difficulty_estimator: Optional[str] = None


class RolloutGenerator:
    """Rollout 生成器。

    负责从模型生成回复，支持批量采样和长尾样本聚焦。
    """

    def __init__(
        self,
        config: RolloutConfig,
        model: torch.nn.Module,
        tokenizer,
        device: Optional[str] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[dict]:
        """为一批 prompt 生成回复。"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "num_return_sequences": self.config.num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        outputs = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for i, prompt in enumerate(prompts):
            prompt_len = inputs["input_ids"].shape[1]
            response_ids = outputs[i][prompt_len:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            results.append(
                {
                    "prompt": prompt,
                    "prompt_ids": inputs["input_ids"][i],
                    "response_ids": response_ids,
                    "response_text": response_text,
                    "full_ids": outputs[i],
                }
            )

        return results

    def generate_from_dataloader(self, dataloader: DataLoader) -> list[dict]:
        """从 DataLoader 批量生成。"""
        all_results = []
        for batch in dataloader:
            prompts = batch["prompt"] if isinstance(batch, dict) else batch
            results = self.generate(prompts)
            all_results.extend(results)
        return all_results
