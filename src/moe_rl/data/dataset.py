"""基础 RLHF 数据集。"""

from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset


@dataclass
class RLHFDatasetConfig:
    """数据集配置。"""

    dataset_name: str = ""
    split: str = "train"
    prompt_column: str = "prompt"
    response_column: str = "response"
    max_prompt_length: int = 512
    max_response_length: int = 1024
    num_proc: int = 4
    cache_dir: Optional[str] = None


class RLHFDataset(Dataset):
    """基础 RLHF 数据集。

    封装 HuggingFace datasets，提供统一的 prompt/response 接口。
    """

    def __init__(self, config: RLHFDatasetConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        result = {
            "prompt": item.get(self.config.prompt_column, ""),
            "query": item.get(self.config.prompt_column, ""),
        }
        if self.config.response_column in item:
            result["response"] = item[self.config.response_column]
        return result

    def load_from_hf(self, dataset_name: Optional[str] = None, split: Optional[str] = None):
        """从 HuggingFace 加载数据集。"""
        try:
            from datasets import load_dataset

            name = dataset_name or self.config.dataset_name
            split = split or self.config.split
            dataset = load_dataset(
                name,
                split=split,
                cache_dir=self.config.cache_dir,
                num_proc=self.config.num_proc,
            )
            self.data = list(dataset)
        except ImportError:
            raise ImportError("请安装 datasets: pip install datasets")

    def filter_by_length(self, max_prompt_length: Optional[int] = None):
        """按长度过滤。"""
        max_len = max_prompt_length or self.config.max_prompt_length
        if self.tokenizer is not None:
            self.data = [
                item
                for item in self.data
                if len(self.tokenizer.encode(item[self.config.prompt_column])) <= max_len
            ]
        else:
            self.data = [
                item
                for item in self.data
                if len(item[self.config.prompt_column].split()) <= max_len
            ]

    def shuffle(self, seed: int = 42):
        """打乱数据。"""
        import random

        random.seed(seed)
        random.shuffle(self.data)
