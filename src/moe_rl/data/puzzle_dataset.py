"""Puzzle 数据集：ARC、Big-Bench Hard 等。"""

import json
from typing import Optional

from .dataset import RLHFDataset, RLHFDatasetConfig


class PuzzleDataset(RLHFDataset):
    """谜题/推理数据集。

    支持 ARC (Abstraction and Reasoning Corpus)、Big-Bench Hard 等
    抽象推理和谜题类基准。
    """

    SUPPORTED_DATASETS = {
        "arc": "allenai/ai2_arc",
        "bbh": "lukaemon/bbh",
    }

    def __init__(self, config: RLHFDatasetConfig, tokenizer=None):
        super().__init__(config, tokenizer)

    def load_arc(self, subset: str = "ARC-Challenge", split: str = "train"):
        """加载 ARC。"""
        try:
            from datasets import load_dataset

            dataset = load_dataset(self.SUPPORTED_DATASETS["arc"], subset, split=split)
            self.data = []
            for item in dataset:
                choices = item["choices"]
                choice_text = "\n".join(
                    [f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])]
                )
                prompt = f"{item[\'question\']}\n\n{choice_text}\n\nAnswer:"
                self.data.append(
                    {
                        "prompt": prompt,
                        "answer": item["answerKey"],
                        "question": item["question"],
                        "choices": choices,
                    }
                )
        except ImportError:
            raise ImportError("请安装 datasets")

    def load_bbh(self, task: str = "all", split: str = "train"):
        """加载 Big-Bench Hard。"""
        try:
            from datasets import load_dataset

            if task == "all":
                all_data = []
                bbh_tasks = [
                    "boolean_expressions",
                    "causal_judgement",
                    "date_understanding",
                    "disambiguation_qa",
                    "formal_fallacies",
                    "geometric_shapes",
                    "hyperbaton",
                    "logical_deduction_five_objects",
                    "logical_deduction_seven_objects",
                    "logical_deduction_three_objects",
                    "movie_recommendation",
                    "multistep_arithmetic_two",
                    "navigate",
                    "object_counting",
                    "penguins_in_a_table",
                    "reasoning_about_colored_objects",
                    "ruin_names",
                    "salient_translation_error_detection",
                    "snarks",
                    "sports_understanding",
                    "temporal_sequences",
                    "tracking_shuffled_objects_five_objects",
                    "tracking_shuffled_objects_seven_objects",
                    "tracking_shuffled_objects_three_objects",
                    "web_of_lies",
                    "word_sorting",
                ]
                for task_name in bbh_tasks:
                    try:
                        dataset = load_dataset(
                            self.SUPPORTED_DATASETS["bbh"], task_name, split=split
                        )
                        for item in dataset:
                            all_data.append(
                                {
                                    "prompt": item["input"],
                                    "answer": item["target"],
                                    "task": task_name,
                                }
                            )
                    except Exception:
                        continue
                self.data = all_data
            else:
                dataset = load_dataset(self.SUPPORTED_DATASETS["bbh"], task, split=split)
                self.data = [
                    {"prompt": item["input"], "answer": item["target"], "task": task}
                    for item in dataset
                ]
        except ImportError:
            raise ImportError("请安装 datasets")

    def extract_answer(self, response: str) -> Optional[str]:
        """从模型回复中提取答案。"""
        import re

        patterns = [
            r"answer is\s*[:\s]*([A-Z])",
            r"####\s*([A-Za-z0-9\-+., ]+)",
            r"\\boxed\{([^}]+)\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        lines = response.strip().split("\n")
        if lines:
            return lines[-1].strip()
        return None

    def verify_answer(self, predicted: Optional[str], reference: Optional[str]) -> bool:
        """验证答案。"""
        if predicted is None or reference is None:
            return False
        pred = str(predicted).strip().lower()
        ref = str(reference).strip().lower()
        return pred == ref or pred.endswith(ref) or ref.endswith(pred)
