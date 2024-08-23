from abc import ABC, abstractmethod


class EpsilonFunc(ABC):
    """
    Interface for a function to compute the current epsilon value.
    """

    @abstractmethod
    def compute_epsilon(self, current_block: int, model_block: int) -> float:
        """
        Computes the epsilon value.

        Args:
            current_block: The current block.
            model_block: The block at which the model was submitted.

        Returns:
            The computed epsilon value.
        """
        pass


class FixedEpsilon(EpsilonFunc):
    """
    A fixed epsilon value.
    """

    def __init__(self, epsilon: float):
        """
        Initializes the FixedEpsilon.

        Args:
            epsilon: The fixed epsilon value.
        """
        self.epsilon = epsilon

    def compute_epsilon(self, current_block: int, model_block: int) -> float:
        return self.epsilon


class LinearDecay(EpsilonFunc):
    """
    An epsilon function linearly decays epsilon to a minimum epsilon over a defined number of blocks.
    """

    def __init__(self, start_epsilon: float, end_epsilon: float, decay_blocks: int):
        """
        Initializes the FixedEpsilon.

        Args:
            epsilon: The fixed epsilon value.
        """
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_blocks = decay_blocks

    def compute_epsilon(self, current_block: int, model_block: int) -> float:
        # Find the difference in blocks.
        # In case of metagraph divergance between checking blocks and syncing models default to 0.
        block_difference = max(current_block - model_block, 0)

        # Use a linear progression based on the decay blocks capping at complete decay.
        block_adjustment = min(block_difference / self.decay_blocks, 1)

        epsilon_adjustment = block_adjustment * (self.start_epsilon - self.end_epsilon)

        return self.start_epsilon - epsilon_adjustment
