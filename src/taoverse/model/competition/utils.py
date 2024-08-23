from typing import List, Optional, Tuple

from taoverse.model.competition.data import Competition


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
