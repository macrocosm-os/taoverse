import copy
import pickle
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Set

import bittensor as bt

from taoverse.model.data import EvalResult, ModelMetadata


class ModelTracker:
    """Tracks model information for each miner.

    Thread safe.
    """

    _MAX_EVAL_HISTORY_LEN = 3

    def __init__(
        self,
        max_eval_history_len: int = _MAX_EVAL_HISTORY_LEN,
    ):
        self.max_eval_history_len = max_eval_history_len

        # Create a dict from miner hotkey to model metadata.
        self.miner_hotkey_to_model_metadata_dict = dict()
        self.miner_hotkey_to_eval_results = defaultdict(list)

        # Make this class thread safe because it will be accessed by multiple threads.
        # One for the downloading new models loop and one for the validating models loop.
        self.lock = threading.RLock()

    def save_state(self, filepath):
        """Save the current state to the provided filepath."""

        # Open a writable binary file for pickle.
        with self.lock:
            with open(filepath, "wb") as f:
                pickle.dump(
                    [
                        self.miner_hotkey_to_model_metadata_dict,
                        self.miner_hotkey_to_eval_results,
                    ],
                    f,
                )

    def load_state(self, filepath):
        """Load the state from the provided filepath."""

        # Open a readable binary file for pickle.
        with open(filepath, "rb") as f:
            (
                self.miner_hotkey_to_model_metadata_dict,
                self.miner_hotkey_to_eval_results,
            ) = pickle.load(f)

    def get_miner_hotkey_to_model_metadata_dict(self) -> Dict[str, ModelMetadata]:
        """Returns the mapping from miner hotkey to model metadata."""

        # Return a copy to ensure outside code can't modify the scores.
        with self.lock:
            return copy.deepcopy(self.miner_hotkey_to_model_metadata_dict)

    def get_model_metadata_for_miner_hotkey(
        self, hotkey: str
    ) -> Optional[ModelMetadata]:
        """Returns the model metadata for a given hotkey if any."""

        with self.lock:
            if hotkey in self.miner_hotkey_to_model_metadata_dict:
                return self.miner_hotkey_to_model_metadata_dict[hotkey]
            return None

    def get_eval_results_for_miner_hotkey(self, hotkey: str) -> List[EvalResult]:
        """Returns the latest evaluation results for a miner, ordered by block of eval (oldest first)."""

        with self.lock:
            return copy.deepcopy(self.miner_hotkey_to_eval_results[hotkey])

    def on_hotkeys_updated(self, incoming_hotkeys: Set[str]):
        """Notifies the tracker which hotkeys are currently being tracked on the metagraph."""

        with self.lock:
            existing_hotkeys = set(self.miner_hotkey_to_model_metadata_dict.keys())
            for hotkey in existing_hotkeys - incoming_hotkeys:
                del self.miner_hotkey_to_model_metadata_dict[hotkey]
                del self.miner_hotkey_to_eval_results[hotkey]
                bt.logging.trace(f"Removed outdated hotkey: {hotkey} from ModelTracker")

    def on_miner_model_updated(
        self,
        hotkey: str,
        model_metadata: ModelMetadata,
    ) -> None:
        """Notifies the tracker that a miner has had their associated model updated.

        Args:
            hotkey (str): The miner's hotkey.
            model_metadata (ModelMetadata): The latest model metadata of the miner.
        """
        with self.lock:
            prev_metadata = self.miner_hotkey_to_model_metadata_dict.get(hotkey, None)
            self.miner_hotkey_to_model_metadata_dict[hotkey] = model_metadata

            # If the model was updated, clear the evaluation results since they're no
            # longer relevant.
            if prev_metadata != model_metadata:
                self.miner_hotkey_to_eval_results[hotkey].clear()

            bt.logging.trace(f"Updated Miner {hotkey}. ModelMetadata={model_metadata}.")

    def on_model_evaluated(self, hotkey: str, result: EvalResult) -> None:
        """Notifies the tracker that a model has been evaluated.

        Args:
            hotkey (str): The miner's hotkey.
            result (EvalResult): The evaluation result.
        """
        with self.lock:
            eval_results = self.miner_hotkey_to_eval_results[hotkey]
            eval_results.append(result)
            eval_results.sort(key=lambda x: x.block)

            if len(eval_results) > self.max_eval_history_len:
                eval_results.pop(0)

            bt.logging.trace(f"Updated eval results {hotkey}. EvalResult={result}.")
