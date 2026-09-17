"""奖励函数模块。"""

from abc import ABC, abstractmethod
from typing import Optional


class RewardFunction(ABC):
    """奖励函数基类。"""

    @abstractmethod
    def __call__(self, prompt: str, response: str, **kwargs) -> float:
        """计算奖励。"""
        raise NotImplementedError

    def batch_compute(self, prompts: list[str], responses: list[str], **kwargs) -> list[float]:
        """批量计算奖励。"""
        return [self(p, r, **kwargs) for p, r in zip(prompts, responses)]


class CodeReward(RewardFunction):
    """代码生成奖励函数。基于测试用例通过率计算奖励。"""

    def __init__(
        self,
        timeout: int = 10,
        reward_per_test: float = 1.0,
        penalty_per_error: float = 0.0,
    ):
        self.timeout = timeout
        self.reward_per_test = reward_per_test
        self.penalty_per_error = penalty_per_error

    def __call__(self, prompt: str, response: str, **kwargs) -> float:
        """计算代码奖励。"""
        test_cases = kwargs.get("test_cases", [])
        if not test_cases:
            return 0.0
        return 0.0

    def _extract_code(self, response: str) -> str:
        """从回复中提取代码。"""
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        return response.strip()


class MathReward(RewardFunction):
    """数学推理奖励函数。基于最终答案正确性计算奖励。"""

    def __init__(
        self,
        tolerance: float = 1e-4,
        partial_reward: bool = False,
        reasoning_reward_weight: float = 0.0,
    ):
        self.tolerance = tolerance
        self.partial_reward = partial_reward
        self.reasoning_reward_weight = reasoning_reward_weight

    def __call__(self, prompt: str, response: str, **kwargs) -> float:
        """计算数学奖励。"""
        reference = kwargs.get("reference_answer", "")
        if not reference:
            return 0.0

        predicted = self._extract_answer(response)
        if predicted is None:
            return 0.0

        if self._verify_answer(predicted, reference):
            return 1.0
        return 0.0

    def _extract_answer(self, response: str) -> Optional[str]:
        """从回复中提取答案。"""
        import re

        patterns = [
            r"####\s*([\-0-9.,]+)",
            r"\\boxed\{([^}]+)\}",
            r"final answer is\s*[:\s]*([\-0-9.,]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "").strip()
        return None

    def _verify_answer(self, predicted: str, reference: str) -> bool:
        """验证答案。"""
        try:
            pred_val = float(predicted)
            ref_val = float(reference)
            return abs(pred_val - ref_val) <= self.tolerance * max(1.0, abs(ref_val))
        except (ValueError, TypeError):
            return str(predicted).strip() == str(reference).strip()


class FormatReward(RewardFunction):
    """格式奖励函数：鼓励模型按指定格式输出。"""

    def __init__(self, expected_format: str = "general"):
        self.expected_format = expected_format

    def __call__(self, prompt: str, response: str, **kwargs) -> float:
        """计算格式奖励。"""
        score = 0.0

        if len(response) > 10:
            score += 0.1

        if "####" in response or "\\boxed" in response or "final answer" in response.lower():
            score += 0.2

        if self.expected_format == "code" and "```" in response:
            score += 0.2

        return min(1.0, score)
