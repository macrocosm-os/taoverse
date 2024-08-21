from typing import List, Optional, Tuple

from taoverse.model.competition.data import Competition, EpsilonDecay


def get_competition_for_block(
    comp_id: int,
    block: int,
    schedule_by_block: List[Tuple[int, List[Competition]]],
) -> Optional[Competition]:
    """Returns the competition for the given id at the given block, or None if it does not exist."""
    competition_schedule = get_competition_schedule_for_block(block, schedule_by_block)
    for comp in competition_schedule:
        if comp.id == comp_id:
            return comp
    return None


def get_competition_schedule_for_block(
    block: int, schedule_by_block: List[Tuple[int, List[Competition]]]
) -> List[Competition]:
    """Returns the competition schedule at block."""
    competition_schedule = None
    for b, schedule in schedule_by_block:
        if block >= b:
            competition_schedule = schedule
    assert (
        competition_schedule is not None
    ), f"No competition schedule found for block {block}"
    return competition_schedule


def get_epsilon_for_earlier_model(
    current_block: int, earlier_model_block: int, epsilon_decay: EpsilonDecay
) -> float:
    """Calculates epsilon value to use for the earlier model in a comparison based on epsilon decay."""

    # Find the difference in blocks.
    # In case of metagraph divergance between checking blocks and syncing models defult to 0.
    block_difference = max(current_block - earlier_model_block, 0)

    # Use a linear progression based on the decay blocks capping at complete decay.
    block_adjustment = min(block_difference / epsilon_decay.decay_blocks, 1)

    epsilon_adjustment = block_adjustment * (
        epsilon_decay.starting_epsilon - epsilon_decay.ending_epsilon
    )

    return epsilon_decay.starting_epsilon - epsilon_adjustment
