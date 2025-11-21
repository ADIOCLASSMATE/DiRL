import os
import re
from typing import TYPE_CHECKING, Dict, List, Union
from utils.math_utils import extract_last_boxed, latex_eval, normalize_final_answer, equation
import json

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, response_token_ids=None, max_tokens=None, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for idx, (content, sol) in enumerate(zip(completions, solution)):
            # 如果超过max_tokens，不管对错，都赋值为错误
            if response_token_ids is not None and max_tokens is not None:
                if len(response_token_ids[idx]) >= max_tokens:
                    print("截断的text为: ...", content[-10:])
                    rewards.append(0.0)
                    continue
            
            extract_answer = extract_last_boxed(content)
            print(extract_answer)
            if extract_answer is None:
                reward = 0.0
            else:
                answer = "\\boxed{"+extract_answer+"}"
                reward = float(equation(answer, sol))
            rewards.append(reward)
        return rewards


class BinaryReward(ORM):
    """Returns 1 for correct answers and -1 for incorrect answers."""
    
    def __init__(self, accuracy_orm=None):
        self.accuracy_orm = accuracy_orm or MathAccuracy()
    
    def __call__(self, completions, solution, response_token_ids=None, max_tokens=None, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, response_token_ids, max_tokens, **kwargs)
        return [1.0 if acc_reward >= 1.0 else -1.0 for acc_reward in acc_rewards]


class Format(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]

class ThinkFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        completions = ["<think>"+x for x in completions]
        pattern = r'^<think>.*?</think>\s*(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class Boxed(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        extracted_answers = [extract_last_boxed(completion) for completion in completions]
        return [0.1 if answer is not None else 0.0 for answer in extracted_answers]

class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        # print(response_token_ids, acc_rewards)
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = min(len(ids), self.max_len)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            if gen_len >= self.max_len:
                reward = self.min_len_value_wrong
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


orms = {
    'accuracy': MathAccuracy,
    'binary': BinaryReward,
    'format': Format,
    'think_format': ThinkFormat,
    'boxed': Boxed,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
