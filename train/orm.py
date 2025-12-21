import os
import re
from typing import TYPE_CHECKING, Dict, List, Union
try:
    from utils.math_utils import extract_last_boxed, latex_eval, normalize_final_answer, equation
except:
    print("math_utils not found, using default extract_last_boxed")
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

    def __call__(self, completions, solution, response_token_ids=None, max_tokens=None, block_size=None, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for idx, (content, sol) in enumerate(zip(completions, solution)):
            # # 如果超过max_tokens或在截断区域，不管对错，都赋值为错误
            # 由于prompt/response边界调整可能移动最多block_size-1个token，所以阈值需要扩大
            if response_token_ids is not None and max_tokens is not None:
                threshold = max_tokens - (block_size if block_size else 0)
                if len(response_token_ids[idx]) >= threshold:
                    # print("截断的text为: ...", content[-10:])
                    rewards.append(0.0)
                    continue
            extract_answer = extract_last_boxed(content)
            # print(extract_answer)
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
    
    def __call__(self, completions, solution, response_token_ids=None, max_tokens=None, block_size=None, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, response_token_ids, max_tokens, block_size, **kwargs)
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
        completions = ["<think>"+x if not x.startswith("<think>") else x for x in completions]
        # print(completions)
        pattern = r'^\s*<think>.*?</think>\s*$'
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


class SpeedReward(ORM):
    """
    Speed-based reward function using step_map.
    Speed metric: response_length / len(set(step_map))
    Range: [1, block_size]
    - Higher metric = faster generation (coarse-grained thinking)
    - Lower metric = slower generation (fine-grained thinking)
    
    Reward strategy:
    - For correct answers: higher speed => higher reward (efficiency is good)
    - For incorrect answers: lower speed => higher reward (need more careful thinking)
    """
    def __init__(self,
                 speed_min_speed_value_wrong: float = 0.0,
                 speed_max_speed_value_wrong: float = -0.5,
                 speed_min_speed_value_correct: float = 0.5,
                 speed_max_speed_value_correct: float = 1.0,
                 speed_max_len: int = 4,
                 accuracy_orm=None):
        """
        Args:
            speed_min_speed_value_wrong: Reward when speed=min (slow) and answer is wrong
            speed_max_speed_value_wrong: Reward when speed=max (fast) and answer is wrong
            speed_min_speed_value_correct: Reward when speed=min (slow) and answer is correct
            speed_max_speed_value_correct: Reward when speed=max (fast) and answer is correct
            speed_max_len: Maximum speed value (should be block_size, e.g., 4, 64)
            accuracy_orm: ORM to determine correctness
        """
        self.min_speed_value_wrong = speed_min_speed_value_wrong
        self.max_speed_value_wrong = speed_max_speed_value_wrong
        self.min_speed_value_correct = speed_min_speed_value_correct
        self.max_speed_value_correct = speed_max_speed_value_correct
        self.max_len = speed_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        step_maps = kwargs.get('step_map', None)
        response_token_ids = kwargs.get('response_token_ids', None)
        
        if step_maps is None or response_token_ids is None:
            # If no step_map or response_token_ids provided, return accuracy rewards only
            return acc_rewards
        
        rewards = []
        for step_map, token_ids, acc_reward in zip(step_maps, response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            
            # Calculate speed metric: response_length / len(set(step_map))
            # Range: [1, block_size]
            response_length = len(token_ids) if token_ids is not None else 0
            unique_steps = len(set(step_map)) if len(step_map) > 0 else 1
            speed_metric = response_length / unique_steps if unique_steps > 0 else 1
            
            # Normalize to [0, max_len-1] since cosfn expects input starting from 0
            # speed=1 → normalized=0, speed=max_len → normalized=max_len-1
            speed_metric_normalized = max(0, min(speed_metric - 1, self.max_len - 1))
            
            if is_correct:
                # For correct answers: higher speed is better
                # cosfn: t=0 → max_value, t=T → min_value
                # We want: speed=min(t=0) → low reward, speed=max(t=T) → high reward
                # So swap: min_value=high, max_value=low
                min_value = self.max_speed_value_correct  # reward at speed=max
                max_value = self.min_speed_value_correct  # reward at speed=min
            else:
                # For incorrect answers: lower speed is better
                # We want: speed=min(t=0) → high reward, speed=max(t=T) → low reward
                # So swap: min_value=low, max_value=high
                min_value = self.max_speed_value_wrong  # reward at speed=max
                max_value = self.min_speed_value_wrong  # reward at speed=min
            
            reward = self.cosfn(speed_metric_normalized, self.max_len - 1, min_value, max_value)
            rewards.append(reward)
        return rewards


class SpeedPenalty(ORM):
    """
    One-sided speed penalty: only penalize fast generation when wrong, give base reward when correct.
    Speed metric: response_length / len(set(step_map))
    Range: [1, block_size]
    
    Reward strategy:
    - For correct answers: always 0.5 (base reward, no speed bonus)
    - For incorrect answers: lower speed => less penalty (need more careful thinking)
    """
    def __init__(self,
                 speed_correct_reward: float = 0.5,
                 speed_min_speed_value_wrong: float = 0.0,
                 speed_max_speed_value_wrong: float = -0.5,
                 speed_max_len: int = 4,
                 accuracy_orm=None):
        """
        Args:
            speed_correct_reward: Base reward when answer is correct (no speed bonus)
            speed_min_speed_value_wrong: Penalty when speed=min (slow) and answer is wrong
            speed_max_speed_value_wrong: Penalty when speed=max (fast) and answer is wrong
            speed_max_len: Maximum speed value (should be block_size)
            accuracy_orm: ORM to determine correctness
        """
        self.correct_reward = speed_correct_reward
        self.min_speed_value_wrong = speed_min_speed_value_wrong
        self.max_speed_value_wrong = speed_max_speed_value_wrong
        self.max_len = speed_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        step_maps = kwargs.get('step_map', None)
        response_token_ids = kwargs.get('response_token_ids', None)
        
        if step_maps is None or response_token_ids is None:
            # If no step_map or response_token_ids provided, return accuracy rewards only
            return acc_rewards
        
        rewards = []
        for step_map, token_ids, acc_reward in zip(step_maps, response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            
            if is_correct:
                # For correct answers: base reward (no speed bonus)
                reward = self.correct_reward
            else:
                # For incorrect answers: penalize fast generation
                # Calculate speed metric: response_length / len(set(step_map))
                response_length = len(token_ids) if token_ids is not None else 0
                unique_steps = len(set(step_map)) if len(step_map) > 0 else 1
                speed_metric = response_length / unique_steps if unique_steps > 0 else 1
                
                # Normalize to [0, max_len-1]
                speed_metric_normalized = max(0, min(speed_metric - 1, self.max_len - 1))
                
                # Lower speed is better for wrong answers
                # cosfn: t=0 → max_value, t=T → min_value
                # We want: speed=min(t=0) → high reward, speed=max(t=T) → low reward
                min_value = self.max_speed_value_wrong  # penalty at speed=max
                max_value = self.min_speed_value_wrong  # penalty at speed=min
                
                reward = self.cosfn(speed_metric_normalized, self.max_len - 1, min_value, max_value)
            
            rewards.append(reward)
        return rewards


class LengthReward(ORM):
    """
    Simple length reward: longer responses get higher rewards.
    If the response is truncated, reward is 0.
    """
    def __init__(self, max_reward: float = 1.0):
        """
        Args:
            max_reward: Maximum reward value for length
        """
        self.max_reward = max_reward
    
    def __call__(self, completions, response_token_ids=None, max_tokens=None, block_size=None, **kwargs) -> List[float]:
        if response_token_ids is None or max_tokens is None:
            # If no token information provided, return zero rewards
            return [0.0] * len(completions)
        
        rewards = []
        for idx, token_ids in enumerate(response_token_ids):
            # Check if truncated
            threshold = max_tokens - (block_size if block_size else 0)
            if len(token_ids) >= threshold:
                # Truncated, reward is 0
                rewards.append(0.0)
            else:
                # Not truncated, reward is proportional to length
                # Normalize by max_tokens to get value in [0, 1], then scale by max_reward
                reward = (len(token_ids) / max_tokens) * self.max_reward
                rewards.append(reward)
        
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
    'speed': SpeedReward,
    'speed_penalty': SpeedPenalty,
    'length': LengthReward,
}

if __name__ == "__main__":
    orm = orms['think_format']()
    completions = ["\nOkay, so I need to convert the point (0,3) from rectangular coordinates to polar coordinates. Hmm, right, I remember that in polar coordinates, each point is represented by an ordered pair (r, θ), where r is the distance from the origin to the point, and θ is the angle from the positive x-axis to the line connecting the origin to the point. \n\nFirst, let me recall the formulas for converting from rectangular to polar coordinates. I think r is calculated using the Pythagorean theorem: r = sqrt(x² + y²). And the angle θ is found using the arctangent function: θ = arctan(y/x). But wait, arctan(y/x) sometimes gives angles in the wrong quadrant, so I have to consider the signs of the point's coordinates to determine the correct quadrant. \n\nIn this case, the point is (0,3). So x is 0 and y is 3. Let's calculate r first. Plugging into the formula: r = sqrt(0² + 3²) = sqrt(0 + 9) = sqrt(9) = 3. So r is 3. That makes sense because the point is on the positive y-axis, 3 units away from the origin.\n\nNow, the angle θ. Since x is 0, θ = arctan(y/x). But here x is 0. Arctangent is usually defined for the tangent of θ, which is opposite over adjacent. However, if x is 0, the tangent is undefined. That means the angle is either π/2 or 3π/2, depending on the sign of y. Since y is positive (3), the angle should be π/2. Let me confirm that. \n\nIf the point is on the positive y-axis, the angle from the positive x-axis is 90 degrees, which is π/2 radians. So θ = π/2. \n\nTherefore, the polar coordinates should be (3, π/2). Let me just double-check. \n\nTo verify, converting back to rectangular coordinates from polar coordinates should give me the original point. The formulas for that are x = r cos θ and y = r sin θ. So if we take r = 3 and θ = π/2, then x = 3 cos(π/2) = 3*0 = 0, and y = 3 sin(π/2) = 3*1 = 3. That matches the original point (0,3). \n\nI think that's right. But just to make sure I didn't make any mistakes, let me consider another approach. If the point is on the positive y-axis, it's in the first quadrant. The angle θ is measured from the positive x-axis. So starting from the positive x-axis, turning counterclockwise to the positive y-axis is 90 degrees, which is π/2 radians. So yes, θ = π/2. \n\nAlternatively, imagine using the arctangent function. Since x is 0, arctan(y/x) is arctan(3/0). But dividing by zero is undefined, which again tells us that the angle is π/2. \n\nAnother thought: sometimes when both x and y are zero, r would be zero, but here r is 3, so that's fine. \n\nTherefore, all checks seem out. The polar coordinates are (3, π/2). \n\n**Final Answer**\nThe polar coordinates are \\boxed{\\left(3, \\dfrac{\\pi}{2}\\right)}.\n </think> \n\nTo convert the point \\((0, 3)\\) from rectangular coordinates to polar coordinates, we start by calculating the radius \\(r\\) using the Pythagorean theorem:\n\n\\[\nr = \\sqrt{x^2 + y^2} = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\n\\]\n\nNext, we determine the angle \\(\\theta\\). Since the point \\((0, 3)\\) lies on the positive y-axis, the angle \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) radians. This is because the angle from the positive x-axis to the positive y-axis is 90 degrees, which is \\(\\frac{\\pi}{2}\\) radians.\n\nTo confirm, we can verify the conversion back to rectangular coordinates using the formulas \\(x = r \\cos \\theta\\) and \\(y = r \\sin \\theta\\):\n\n\\[\nx = 3 \\cos \\left(\\frac{\\pi}{2}\\right) = 3 \\cdot 0 = 0\n\\]\n\\[\ny = 3 \\sin \\left(\\frac{\\pi}{2}\\right) = 3 \\cdot 1 = 3\n\\]\n\nThese calculations match the original point \\((0, 3)\\), confirming the correctness of the polar coordinates.\n\nThus, the polar coordinates are \\(\\boxed{\\left(3, \\dfrac{\\pi}{2}\\right)}\\). <|im_end|>"]
    print(orm(completions))
