from typing import List, Set, Tuple

import bittensor as bt


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph) -> int:
    """Asserts the wallet is a registered miner and returns the miner's UID.

    Raises:
        ValueError: If the wallet is not registered.
    """
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(
            f"You are not registered. \nUse: \n`btcli s register --netuid {metagraph.netuid}` to register via burn \n or btcli s pow_register --netuid {metagraph.netuid} to register with a proof of work"
        )
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.success(
        f"You are registered with address: {wallet.hotkey.ss58_address} and uid: {uid}"
    )

    return uid


def get_top_miners(
    metagraph: bt.metagraph, min_vali_stake: int, min_miner_weight_percent: float
) -> Set[int]:
    """Returns the set of top miners, chosen based on weights set on the valis above the specifed threshold.

    Args:
        metagraph (bt.metagraph): Metagraph to use. Must not be lite.
        min_vali_stake (int): Minimum stake threshold for a vali's weights to be considered.
        min_miner_weight_percent (float): Minimum weight on a vali for the miner to count as a top miner.
    """

    top_miners = set()

    # Find validators over 100k in stake.
    valis_by_stake = get_high_stake_validators(metagraph, min_vali_stake)

    # For each, find miners with at least min_miner_weight_percent of the weights.
    # Since there can be multiple competitions at different reward percentages we can't just check biggest.
    for uid in valis_by_stake:
        # Weights is a list of (uid, weight) pairs
        weights: List[Tuple[int, float]] = metagraph.neurons[uid].weights
        total_weight = sum(weight for _, weight in weights)

        threshold = total_weight * min_miner_weight_percent
        for uid, weight in weights:
            if weight >= threshold:
                top_miners.add(uid)

    return top_miners


def get_high_stake_validators(metagraph: bt.metagraph, min_stake: int) -> Set[int]:
    """Returns a set of validators at or above the specified stake threshold for the subnet"""
    valis = set()

    for uid, stake in enumerate(metagraph.S):
        # Use vPermit to check for validators rather than vTrust because we'd rather
        # cast a wide net in the case that vTrust is 0 due to an unhealthy state of the
        # subnet.
        if stake >= min_stake and metagraph.validator_permit[uid]:
            valis.add(uid)

    return valis


def get_hash_of_sync_block(subtensor: bt.subtensor, sync_cadence: int) -> int:
    """Returns the hash of the most recent block that is a multiple of the sync cadence."""
    current_block = subtensor.get_current_block()
    sync_block = current_block // sync_cadence * sync_cadence
    return hash(subtensor.get_block_hash(sync_block))
