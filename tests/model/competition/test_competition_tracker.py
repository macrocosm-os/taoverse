import os
import unittest

import torch
from transformers import LlamaForCausalLM

from taoverse.model.competition.competition_tracker import CompetitionTracker
from taoverse.model.competition.data import Competition, ModelConstraints
from taoverse.model.competition.epsilon import FixedEpsilon


class TestCompetitionTracker(unittest.TestCase):
    COMPETITION_1_PARAMETERS = Competition(
        id=1,
        constraints=ModelConstraints(
            max_model_parameter_size=8 * 1024 * 1024 * 1024,
            sequence_length=4096,
            allowed_architectures=[LlamaForCausalLM],
            tokenizer="Xenova/gpt-4",
            kwargs={
                "torch_dtype": torch.bfloat16,
            },
            eval_block_delay=1200,  # ~4 hours.
            epsilon_func=FixedEpsilon(0.005),
            max_bytes=15 * 1024 * 1024 * 1024,  # 15 GB
        ),
        reward_percentage=0.6,
    )
    COMPETITION_2_PARAMETERS = Competition(
        id=2,
        constraints=ModelConstraints(
            max_model_parameter_size=2 * 1024 * 1024 * 1024,
            sequence_length=2048,
            allowed_architectures=[LlamaForCausalLM],
            tokenizer="Xenova/gpt-4",
            kwargs={
                "torch_dtype": torch.bfloat16,
            },
            eval_block_delay=1200,  # ~4 hours.
            epsilon_func=FixedEpsilon(0.005),
            max_bytes=15 * 1024 * 1024 * 1024,  # 15 GB
        ),
        reward_percentage=0.4,
    )

    def setUp(self):
        self.num_neurons = 4
        self.alpha = 0.5
        self.competition_tracker = CompetitionTracker(self.num_neurons, self.alpha)

    def test_record_competition_weights_new(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )

        # Since this is a net new competition, check that weights go immediately to the new values.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[1],
                torch.Tensor([0.1, 0.2, 0.3, 0.4]),
            )
        )

    def test_record_competition_weights_normalized(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0.2, 0.4, 0.6, 0.8])
        )

        # Check that the weights are normalized to sum to 1.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[1],
                torch.Tensor([0.1, 0.2, 0.3, 0.4]),
            )
        )

    def test_record_competition_weights_moving_average(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([1, 0, 0, 0])
        )
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0, 0, 0, 1])
        )

        # Check that the weights are a moving average according to the alpha of 0.5.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[1],
                torch.Tensor([0.5, 0, 0, 0.5]),
            )
        )

    def test_get_competition_weights(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )

        weights = self.competition_tracker.get_competition_weights(1)

        self.assertTrue(torch.equal(weights, torch.Tensor([0.1, 0.2, 0.3, 0.4])))

    def test_get_subnet_weights_one_competition(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )

        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS]
        )

        self.assertTrue(torch.equal(weights, torch.Tensor([0.1, 0.2, 0.3, 0.4])))

    def test_get_subnet_weights_one_competition_with_threshold(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0.1, 0.1, 0.2, 0.3, 0.3])
        )

        # Only count uids with at least 0.3 weight for this competition.
        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS], min_comp_weight_threshold=0.3
        )

        # On normalization expect the two remaining 0.3s to split the weight evenly.
        self.assertTrue(torch.equal(weights, torch.Tensor([0.0, 0.0, 0.0, 0.5, 0.5])))

    def test_get_subnet_weights_two_competitions(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([1, 1, 0, 0])
        )
        self.competition_tracker.record_competition_weights(
            2, torch.Tensor([0, 0, 5, 5])
        )

        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS, self.COMPETITION_2_PARAMETERS]
        )

        # Check that the weights are both normalized and rewarded according to competition reward percent.
        self.assertTrue(torch.equal(weights, torch.Tensor([0.3, 0.3, 0.2, 0.2])))

    def test_get_subnet_weights_two_competitions_with_threshold(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([1, 2, 0, 0])
        )
        self.competition_tracker.record_competition_weights(
            2, torch.Tensor([0, 0, 5, 5])
        )

        # After normalization only the 2nd uid from comp 1 should have any weight.
        weights = self.competition_tracker.get_subnet_weights(
            [self.COMPETITION_1_PARAMETERS, self.COMPETITION_2_PARAMETERS],
            min_comp_weight_threshold=0.4,
        )

        # Check that the weights are both normalized and rewarded according to competition reward percent.
        # Although only comp 1 was trimmed the remaining uid should still receive the full 60%.
        self.assertTrue(torch.equal(weights, torch.Tensor([0.0, 0.6, 0.2, 0.2])))

    def test_resize_one_competition(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0.1, 0.2, 0.3, 0.2, 0.2])
        )

        # Check that the internal state immediately expands to 5 neurons.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[1],
                torch.Tensor(
                    [0.1, 0.2, 0.3, 0.2, 0.2],
                ),
            )
        )

    def test_resize_two_competition(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([0.1, 0.2, 0.3, 0.4])
        )
        self.competition_tracker.record_competition_weights(
            2, torch.Tensor([0.1, 0.2, 0.3, 0.2, 0.2])
        )

        # Check that the internal state of the first competition is expanded with 0s.
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[1],
                torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[2],
                torch.Tensor([0.1, 0.2, 0.3, 0.2, 0.2]),
            )
        )

    def test_reset_competitions(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([1, 0, 0, 0])
        )
        self.competition_tracker.record_competition_weights(
            2, torch.Tensor([0, 0, 0, 1])
        )

        self.competition_tracker.reset_competitions({1})

        # Check that the weights for competition 2 are no longer tracked.
        self.assertTrue(1 in self.competition_tracker.weights_by_competition)
        self.assertFalse(2 in self.competition_tracker.weights_by_competition)

    def test_roundtrip_state(self):
        self.competition_tracker.record_competition_weights(
            1, torch.Tensor([1, 0, 0, 0])
        )

        state_path = ".test_competition_tracker_state.pickle"
        self.competition_tracker.save_state(state_path)

        new_competition_tracker = CompetitionTracker(num_neurons=0)
        new_competition_tracker.load_state(state_path)

        os.remove(state_path)

        self.assertEqual(
            self.competition_tracker.num_neurons,
            new_competition_tracker.num_neurons,
        )
        self.assertEqual(
            self.competition_tracker.alpha,
            new_competition_tracker.alpha,
        )
        self.assertEqual(
            len(new_competition_tracker.weights_by_competition),
            1,
        )
        self.assertTrue(
            torch.equal(
                self.competition_tracker.weights_by_competition[1],
                new_competition_tracker.weights_by_competition[1],
            )
        )


if __name__ == "__main__":
    unittest.main()
