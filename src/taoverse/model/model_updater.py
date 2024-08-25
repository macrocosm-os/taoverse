import statistics
from typing import Dict, List, Optional, Tuple

import bittensor as bt

from taoverse.model.competition import utils as competition_utils
from taoverse.model.competition.data import Competition, ModelConstraints
from taoverse.model.data import Model, ModelMetadata
from taoverse.model.model_tracker import ModelTracker
from taoverse.model.storage.local_model_store import LocalModelStore
from taoverse.model.storage.model_metadata_store import ModelMetadataStore
from taoverse.model.storage.remote_model_store import RemoteModelStore
from taoverse.model.utils import get_hash_of_two_strings


class MinerMisconfiguredError(Exception):
    """Error raised when a miner is misconfigured."""

    def __init__(self, hotkey: str, message: str):
        self.hotkey = hotkey
        super().__init__(f"[{hotkey}] {message}")


class ModelUpdater:
    """Checks if the currently tracked model for a hotkey matches what the miner committed to the chain."""

    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker

    @staticmethod
    def verify_model_satisfies_parameters(
        model: Model, model_constraints: ModelConstraints
    ) -> bool:
        if not model_constraints:
            bt.logging.trace(f"No competition found for {model.id.competition_id}")
            return False

        # Check that the parameter count of the model is within allowed bounds.
        parameter_size = sum(p.numel() for p in model.pt_model.parameters())

        if (
            parameter_size > model_constraints.max_model_parameter_size
            or parameter_size < model_constraints.min_model_parameter_size
        ):
            bt.logging.debug(
                f"Model {model.id.name} does not satisfy constraints for competition {model.id.competition_id}"
            )
            bt.logging.debug(f"Number of model parameters is {parameter_size}")
            bt.logging.debug(
                f"Max parameters allowed is {model_constraints.max_model_parameter_size}"
            )
            bt.logging.debug(
                f"Min parameters allowed is {model_constraints.min_model_parameter_size}"
            )
            return False

        # Make sure it's an allowed architecture.
        if type(model.pt_model) not in model_constraints.allowed_architectures:
            return False

        # Check parameters are sane if specified
        if model_constraints.norm_validation_constraints is not None:
            return ModelUpdater._validate_parameters(
                model.pt_model,
                model_constraints.norm_validation_constraints.norm_eps_soft,
                model_constraints.norm_validation_constraints.norm_eps_soft_percent_threshold,
                model_constraints.norm_validation_constraints.norm_eps_hard,
            )

        return True

    async def _get_metadata(self, hotkey: str) -> Optional[ModelMetadata]:
        """Get metadata about a model by hotkey"""
        return await self.metadata_store.retrieve_model_metadata(hotkey)

    async def sync_model(
        self,
        hotkey: str,
        curr_block: int,
        schedule_by_block: List[Tuple[int, List[Competition]]],
        force: bool = False,
    ) -> bool:
        """Updates local model for a hotkey if out of sync and returns if it was updated."

        Args:
           hotkey (str): The hotkey of the model to sync.
           curr_block (int): The current block.
           force (bool): Whether to force a sync for this model, even if it's chain metadata hasn't changed.
           schedule_by_block (List[Tuple[int, List[Competition]]]): Which competitions are being run at a given block.
        """
        # Get the metadata for the miner.
        metadata = await self._get_metadata(hotkey)

        if not metadata:
            raise MinerMisconfiguredError(
                hotkey, f"No valid metadata found on the chain"
            )

        # Check that the metadata indicates a competition available at time of upload.
        competition = competition_utils.get_competition_for_block(
            comp_id=metadata.id.competition_id,
            block=metadata.block,
            schedule_by_block=schedule_by_block,
        )
        if not competition:
            raise MinerMisconfiguredError(
                hotkey,
                f"No competition found for {metadata.id.competition_id} at block {metadata.block}",
            )

        # Check that the metadata is old enough to meet the eval_block_delay for the competition.
        # If not we return false and will check again next time we go through the update loop.
        if curr_block - metadata.block < competition.constraints.eval_block_delay:
            bt.logging.debug(
                f"""Sync for hotkey {hotkey} delayed as the current block: {curr_block} is not at least 
                {competition.constraints.eval_block_delay} blocks after the upload block: {metadata.block}. 
                Will automatically retry later."""
            )
            return False

        # Check what model id the model tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )
        # If we are not forcing a sync due to retrying a top model we can short-circuit if no change.
        if not force and metadata == tracker_model_metadata:
            return False

        # Get the local path based on the local store to download to (top level hotkey path)
        path = self.local_store.get_path(hotkey)

        # Otherwise we need to download the new model based on the metadata.
        model = await self.remote_store.download_model(
            metadata.id, path, competition.constraints
        )

        # Update the tracker even if the model fails the following checks to avoid redownloading without new metadata.
        self.model_tracker.on_miner_model_updated(hotkey, metadata)

        # Check that the hash of the downloaded content matches.
        # This is only useful for SN9's legacy competition before multi-competition support
        # was introduced. Securing hashes was optional. In modern competitions, `hash` is
        # always None, and only `secure_hash` is used.
        if model.id.hash != metadata.id.hash:
            # Check that the hash of the downloaded content matches.
            secure_hash = get_hash_of_two_strings(model.id.hash, hotkey)
            if secure_hash != metadata.id.secure_hash:
                raise MinerMisconfiguredError(
                    hotkey,
                    f"Hash of content downloaded from hugging face does not match chain metadata. {metadata}",
                )

        if not ModelUpdater.verify_model_satisfies_parameters(
            model, competition.constraints
        ):
            raise MinerMisconfiguredError(
                hotkey,
                f"Model does not satisfy parameters for competition {competition.id}",
            )

        return True

    @staticmethod
    def _validate_parameters(
        base_model, eps_soft, eps_soft_percent_threshold, eps_hard, print_vals=False
    ) -> bool:
        """
        Validate that parameters of a model are below the specified epsilon limits.

        Parameters:
            base_model (transformers.PreTrainedModel): The base model instance.
            num_layers (int): Number of layers in the model to inspect.
            eps_soft (float): Calculate the percentage of layers above this norm
            eps_soft_percent_threshold (float): Threshold of percentage above eps_soft that will trigger a detection
            eps_hard (float): Hard limit for any norm
        """

        exceed_counts = {
            "q_proj": 0,
            "k_proj": 0,
            "v_proj": 0,
            "o_proj": 0,
            "up_proj": 0,
            "down_proj": 0,
        }
        total_counts = {
            "q_proj": 0,
            "k_proj": 0,
            "v_proj": 0,
            "o_proj": 0,
            "up_proj": 0,
            "down_proj": 0,
        }
        if print_vals:
            avg_norms = {
                "q_proj": 0.0,
                "k_proj": 0.0,
                "v_proj": 0.0,
                "o_proj": 0.0,
                "up_proj": 0.0,
                "down_proj": 0.0,
            }
            max_norms = {
                "q_proj": 0.0,
                "k_proj": 0.0,
                "v_proj": 0.0,
                "o_proj": 0.0,
                "up_proj": 0.0,
                "down_proj": 0.0,
            }

        for layer in base_model.model.layers:
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                weight_norm = getattr(layer.self_attn, proj).weight.norm().item()
                if weight_norm > eps_hard:
                    return False
                elif weight_norm > eps_soft:
                    exceed_counts[proj] += 1
                total_counts[proj] += 1
                if print_vals:
                    avg_norms[proj] += weight_norm
                    max_norms[proj] = max(max_norms[proj], weight_norm)

            # up_proj and down_proj are in the mlp layer
            for proj in ["up_proj", "down_proj"]:
                weight_norm = getattr(layer.mlp, proj).weight.norm().item()
                if weight_norm > eps_hard:
                    return False
                elif weight_norm > eps_soft:
                    exceed_counts[proj] += 1
                total_counts[proj] += 1
                if print_vals:
                    avg_norms[proj] += weight_norm
                    max_norms[proj] = max(max_norms[proj], weight_norm)

        # Calculating and printing percentages
        percentages = [
            exceed_counts[proj] / total_counts[proj] for proj in exceed_counts
        ]

        if print_vals:
            for key, value in total_counts.items():
                avg_norms[key] = avg_norms[key] / value
            print(avg_norms)
            print(max_norms)
            print(percentages)

        return statistics.fmean(percentages) <= eps_soft_percent_threshold
