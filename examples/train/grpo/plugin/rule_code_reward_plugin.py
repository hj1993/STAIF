from typing import List

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


class DatasetRuleCodeReward(ORM):

    def __init__(self, rule_reward_aggregation: str = 'mean'):
        assert rule_reward_aggregation in {'mean', 'all_or_nothing'}
        self.rule_reward_aggregation = rule_reward_aggregation

    @staticmethod
    def _get_check_fn(code_str: str):
        namespace = {}
        exec(code_str, namespace)
        check_fn = namespace.get('check_following')
        if check_fn is None or not callable(check_fn):
            raise ValueError('Each code snippet must define a callable check_following(response) function.')
        return check_fn

    def __call__(self, completions, code, **kwargs) -> List[float]:
        rewards = []
        for completion, code_obj in zip(completions, code):
            if not isinstance(code_obj, dict):
                rewards.append(0.0)
                continue

            code_list = code_obj.get('code') or []
            if not code_list:
                rewards.append(0.0)
                continue

            per_rule_scores = []
            for code_str in code_list:
                try:
                    check_fn = self._get_check_fn(code_str)
                    passed = bool(check_fn(completion))
                    per_rule_scores.append(1.0 if passed else 0.0)
                except Exception as e:
                    logger.warning(f'DatasetRuleCodeReward execution failed: {e}')
                    per_rule_scores.append(0.0)

            if self.rule_reward_aggregation == 'mean':
                reward = sum(per_rule_scores) / len(per_rule_scores)
            else:
                reward = 1.0 if all(score == 1.0 for score in per_rule_scores) else 0.0
            rewards.append(reward)
        return rewards


orms['dataset_rule_code_reward'] = DatasetRuleCodeReward
