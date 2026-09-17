"""Math 数据集：GSM8K、MATH、AQuA 等。"""

import re
from typing import Optional

from .dataset import RLHFDataset, RLHFDatasetConfig


class MathDataset(RLHFDataset):
    """数学推理数据集。

    支持 GSM8K、MATH、AQuA、SVAMP 等数学推理基准。
    """

    SUPPORTED_DATASETS = {
        "gsm8k": "gsm8k",
        "math": "hendrycks/competition_math",
        "aqua": "aqua_rat",
        "svamp": "ChilleD/SVAMP",
    }

    def __init__(self, config: RLHFDatasetConfig, tokenizer=None):
        super().__init__(config, tokenizer)
        self.answers = {}

    def load_gsm8k(self, split: str = "train", name: str = "main"):
        """加载 GSM8K。"""
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                self.SUPPORTED_DATASETS["gsm8k"], name, split=split
            )
            self.data = []
            for item in dataset:
                self.data.append(
                    {
                        "prompt": item["question"],
                        "answer": item["answer"],
                    }
                )
                self.answers[item["question"]] = self._extract_gsm8k_answer(
                    item["answer"]
                )
        except ImportError:
            raise ImportError("请安装 datasets")

    def load_math(self, split: str = "train"):
        """加载 MATH。"""
        self.config.dataset_name = self.SUPPORTED_DATASETS["math"]
        self.config.prompt_column = "problem"
        self.load_from_hf(split=split)

    def _extract_gsm8k_answer(self, answer_text: str) -> Optional[str]:
        """从 GSM8K 答案中提取最终数字。"""
        match = re.search(r"####\s*([\-0-9.,]+)", answer_text)
        if match:
            return match.group(1).replace(",", "")
        return None

    def extract_answer(self, response: str) -> Optional[str]:
        """从模型回复中提取最终答案。"""
        patterns = [
            r"####\s*([\-0-9.,]+)",
            r"\\boxed\{([^}]+)\}",
            r"final answer is\s*[:\s]*([\-0-9.,]+)",
            r"答案是\s*[:\s]*([\-0-9.,]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "").strip()
        return None

    def verify_answer(
        self, predicted: Optional[str], reference: Optional[str], tolerance: float = 1e-4
    ) -> bool:
        """验证答案是否正确。"""
        if predicted is None or reference is None:
            return False
        try:
            pred_val = float(predicted)
            ref_val = float(reference)
            return abs(pred_val - ref_val) <= tolerance * max(1.0, abs(ref_val))
        except (ValueError, TypeError):
            return str(predicted).strip() == str(reference).strip()
