import unittest

from taoverse.model.competition.epsilon import FixedEpsilon, LinearDecay


class TestEpsilonFuncs(unittest.TestCase):
    def test_fixed_decay_is_always_fixed(self):
        epsilon_decay = FixedEpsilon(0.005)

        model_block = 100
        current_block = [95, 100, 105]
        for block in current_block:
            self.assertEqual(
                epsilon_decay.compute_epsilon(
                    model_block=model_block, current_block=block
                ),
                0.005,
            )

    def test_linear_decay_current_block_earlier_than_model_block(self):
        epsilon_decay = LinearDecay(
            start_epsilon=0.005,
            end_epsilon=0.001,
            decay_blocks=7200 * 7,
        )

        calculated_epsilon = epsilon_decay.compute_epsilon(
            current_block=0, model_block=7200
        )

        self.assertEqual(calculated_epsilon, epsilon_decay.start_epsilon)

    def test_linear_decay_model_block_is_current_block(self):
        epsilon_decay = LinearDecay(
            start_epsilon=0.005,
            end_epsilon=0.001,
            decay_blocks=7200 * 7,
        )

        calculated_epsilon = epsilon_decay.compute_epsilon(
            current_block=7200, model_block=7200
        )

        self.assertEqual(calculated_epsilon, epsilon_decay.start_epsilon)

    def test_linear_decay_currnet_block_is_middle_block(self):
        epsilon_decay = LinearDecay(
            start_epsilon=0.005,
            end_epsilon=0.001,
            decay_blocks=7200 * 7,
        )

        calculated_epsilon = epsilon_decay.compute_epsilon(
            current_block=epsilon_decay.decay_blocks / 2,
            model_block=0,
        )

        self.assertEqual(
            calculated_epsilon,
            (epsilon_decay.start_epsilon + epsilon_decay.end_epsilon) / 2,
        )

    def test_linear_decay_current_block_is_decay_blocks(self):
        epsilon_decay = LinearDecay(
            start_epsilon=0.005,
            end_epsilon=0.001,
            decay_blocks=7200 * 7,
        )

        calculated_epsilon = epsilon_decay.compute_epsilon(
            current_block=7200 * 8,
            model_block=7200,
        )

        self.assertEqual(calculated_epsilon, epsilon_decay.end_epsilon)

    def test_linear_decay_current_block_beyond_max_block(self):
        epsilon_decay = LinearDecay(
            start_epsilon=0.005,
            end_epsilon=0.001,
            decay_blocks=7200 * 7,
        )

        calculated_epsilon = epsilon_decay.compute_epsilon(
            current_block=7200 * 100,
            model_block=7200,
        )

        self.assertEqual(calculated_epsilon, epsilon_decay.end_epsilon)


if __name__ == "__main__":
    unittest.main()
