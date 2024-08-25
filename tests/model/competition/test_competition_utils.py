import unittest

import torch
from transformers import (
    BartForCausalLM,
    FalconForCausalLM,
    GemmaForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    PhiForCausalLM,
)
from typing import Dict, List, Tuple

from taoverse.model.competition.data import (
    Competition,
    ModelConstraints,
    NormValidationConstraints,
)
from taoverse.model.competition.epsilon import FixedEpsilon
from taoverse.model.competition.utils import (
    get_competition_for_block,
    get_competition_schedule_for_block,
)

unittest.util._MAX_LENGTH = 2000


class TestCompetitionUtils(unittest.TestCase):
    COMPETITION_ID = 1

    MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[int, ModelConstraints] = {
        COMPETITION_ID: ModelConstraints(
            max_model_parameter_size=6_900_000_000,
            sequence_length=4096,
            allowed_architectures=[
                MistralForCausalLM,
                LlamaForCausalLM,
                BartForCausalLM,
                FalconForCausalLM,
                GPTNeoXForCausalLM,
                PhiForCausalLM,
                GemmaForCausalLM,
            ],
            tokenizer="Xenova/gpt-4",
            kwargs={
                "torch_dtype": torch.bfloat16,
            },
            eval_block_delay=1200,  # ~4 hours.
            norm_validation_constraints=NormValidationConstraints(
                norm_eps_soft=200,
                norm_eps_soft_percent_threshold=0.15,
                norm_eps_hard=1000,
            ),
            epsilon_func=FixedEpsilon(0.005),
            max_bytes=15*1024*1024*1024,  # 15 GB
        ),
    }

    COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
        (
            0,
            [
                Competition(
                    COMPETITION_ID,
                    MODEL_CONSTRAINTS_BY_COMPETITION_ID[COMPETITION_ID],
                    1.0,
                )
            ],
        )
    ]

    def test_get_competition_for_block_valid_competition(self):
        expected_competition = Competition(
            id=self.COMPETITION_ID,
            constraints=ModelConstraints(
                max_model_parameter_size=6_900_000_000,
                sequence_length=4096,
                allowed_architectures=[
                    MistralForCausalLM,
                    LlamaForCausalLM,
                    BartForCausalLM,
                    FalconForCausalLM,
                    GPTNeoXForCausalLM,
                    PhiForCausalLM,
                    GemmaForCausalLM,
                ],
                tokenizer="Xenova/gpt-4",
                eval_block_delay=1200,
                kwargs={
                    "torch_dtype": torch.bfloat16,
                },
                norm_validation_constraints=NormValidationConstraints(
                    norm_eps_soft=200,
                    norm_eps_soft_percent_threshold=0.15,
                    norm_eps_hard=1000,
                ),
                epsilon_func=FixedEpsilon(0.005),
                max_bytes=15*1024*1024*1024,  # 15 GB
            ),
            reward_percentage=1.0,
        )

        competition = get_competition_for_block(
            comp_id=self.COMPETITION_ID,
            block=0,
            schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
        )
        self.assertEqual(competition, expected_competition)

    def test_get_competition_for_block_invalid_competition(self):
        competition = get_competition_for_block(
            comp_id=-1, block=0, schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK
        )
        self.assertIsNone(competition)

    def test_get_competition_for_block_invalid_block(self):
        with self.assertRaises(Exception):
            _ = get_competition_for_block(
                comp_id=self.COMPETITION_ID,
                block=-1,
                schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
            )

    def test_get_competition_schedule_for_block_valid_block(self):
        expected_competition_schedule = [
            Competition(
                id=self.COMPETITION_ID,
                constraints=ModelConstraints(
                    max_model_parameter_size=6_900_000_000,
                    sequence_length=4096,
                    allowed_architectures=[
                        MistralForCausalLM,
                        LlamaForCausalLM,
                        BartForCausalLM,
                        FalconForCausalLM,
                        GPTNeoXForCausalLM,
                        PhiForCausalLM,
                        GemmaForCausalLM,
                    ],
                    tokenizer="Xenova/gpt-4",
                    eval_block_delay=1200,
                    kwargs={
                        "torch_dtype": torch.bfloat16,
                    },
                    norm_validation_constraints=NormValidationConstraints(
                        norm_eps_soft=200,
                        norm_eps_soft_percent_threshold=0.15,
                        norm_eps_hard=1000,
                    ),
                    epsilon_func=FixedEpsilon(0.005),
                    max_bytes=15*1024*1024*1024,  # 15 GB
                ),
                reward_percentage=1.0,
            ),
        ]

        competition_schedule = get_competition_schedule_for_block(
            block=0, schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK
        )
        self.assertEqual(competition_schedule, expected_competition_schedule)

    def test_get_competition_schedule_for_block_invalid_block(self):
        with self.assertRaises(Exception):
            _ = get_competition_schedule_for_block(
                block=-1, schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK
            )


if __name__ == "__main__":
    unittest.main()
