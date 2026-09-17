"""Code 数据集：HumanEval、MBPP、CodeContests 等。"""

from typing import Optional

from .dataset import RLHFDataset, RLHFDatasetConfig


class CodeDataset(RLHFDataset):
    """代码生成数据集。

    支持 HumanEval、MBPP、CodeContests、APPS 等代码生成基准。
    """

    SUPPORTED_DATASETS = {
        "humaneval": "openai/openai_humaneval",
        "mbpp": "mbpp",
        "code_contests": "deepmind/code_contests",
        "apps": "codeparrot/apps",
    }

    def __init__(self, config: RLHFDatasetConfig, tokenizer=None):
        super().__init__(config, tokenizer)
        self.test_cases = {}

    def load_humaneval(self, split: str = "test"):
        """加载 HumanEval。"""
        self.config.dataset_name = self.SUPPORTED_DATASETS["humaneval"]
        self.config.prompt_column = "prompt"
        self.load_from_hf(split=split)

        for item in self.data:
            task_id = item.get("task_id", "")
            self.test_cases[task_id] = {
                "test": item.get("test", ""),
                "entry_point": item.get("entry_point", ""),
            }

    def load_mbpp(self, split: str = "train"):
        """加载 MBPP。"""
        self.config.dataset_name = self.SUPPORTED_DATASETS["mbpp"]
        self.config.prompt_column = "text"
        self.load_from_hf(split=split)

    def format_prompt(self, problem: str, language: str = "python") -> str:
        """格式化代码生成 prompt。"""
        if language == "python":
            return f"""Please complete the following Python function.

{problem}

"""
        return problem

    def extract_code(self, response: str) -> str:
        """从模型回复中提取代码。"""
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        return response.strip()

    def evaluate_code(self, code: str, task_id: str, timeout: int = 10) -> dict:
        """执行代码并评估（需要沙箱环境）。"""
        return {"task_id": task_id, "passed": False, "error": "not implemented"}
