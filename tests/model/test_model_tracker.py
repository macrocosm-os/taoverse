import math
import os
import random
import unittest
from dataclasses import replace

from taoverse.model.data import EvalResult, ModelId, ModelMetadata
from taoverse.model.model_tracker import ModelTracker


class TestModelTracker(unittest.TestCase):
    def setUp(self):
        self.model_tracker = ModelTracker()

    def _create_model_id(self) -> ModelId:
        return ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
            secure_hash="test_secure_hash",
            competition_id=1,
        )

    def test_roundtrip_state(self):
        hotkey = "test_hotkey"
        model_id = self._create_model_id()
        model_metadata = ModelMetadata(id=model_id, block=1)

        state_path = ".test_tracker_state.pickle"
        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.on_model_evaluated(
            hotkey,
            1,
            EvalResult(
                block=2, score=math.inf, winning_model_block=1, winning_model_score=2
            ),
        )
        self.model_tracker.on_model_evaluated(
            hotkey,
            1,
            EvalResult(
                block=3, score=0.5, winning_model_block=1, winning_model_score=2
            ),
        )
        self.model_tracker.save_state(state_path)

        new_tracker = ModelTracker()
        new_tracker.load_state(state_path)

        os.remove(state_path)

        self.assertEqual(
            self.model_tracker.miner_hotkey_to_model_metadata_dict,
            new_tracker.miner_hotkey_to_model_metadata_dict,
        )
        self.assertEqual(
            self.model_tracker.miner_hotkey_to_eval_results,
            new_tracker.miner_hotkey_to_eval_results,
        )

    def test_on_miner_model_updated_add(self):
        hotkey = "test_hotkey"
        model_id = self._create_model_id()

        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)

        self.assertTrue(
            hotkey in self.model_tracker.miner_hotkey_to_model_metadata_dict
        )
        self.assertEqual(
            model_metadata,
            self.model_tracker.miner_hotkey_to_model_metadata_dict[hotkey],
        )

    def test_on_miner_model_updated_update(self):
        hotkey = "test_hotkey"
        model_id = self._create_model_id()

        model_metadata = ModelMetadata(id=model_id, block=1)

        new_model_id = replace(
            model_id,
            commit="other_commit",
            hash="other_hash",
            secure_hash="other_secure_hash",
        )
        new_model_metadata = ModelMetadata(id=new_model_id, block=2)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.on_miner_model_updated(hotkey, new_model_metadata)

        self.assertTrue(
            hotkey in self.model_tracker.miner_hotkey_to_model_metadata_dict
        )
        self.assertEqual(
            new_model_metadata,
            self.model_tracker.miner_hotkey_to_model_metadata_dict[hotkey],
        )

    def test_get_model_metadata_for_miner_hotkey(self):
        hotkey = "test_hotkey"
        model_id = model_id = self._create_model_id()

        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        returned_model_metadata = (
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        )

        self.assertEqual(model_metadata, returned_model_metadata)

    def test_get_model_metadata_for_miner_hotkey_optional(self):
        hotkey = "test_hotkey"

        returned_model_id = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )

        self.assertIsNone(returned_model_id)

    def test_get_miner_hotkey_to_model_metadata_dict(self):
        hotkey_1 = "test_hotkey"
        model_id_1 = ModelId(
            namespace="test_model",
            name="test_name",
            commit="test_commit",
            hash="test_hash",
            secure_hash="test_secure_hash",
            competition_id=1,
        )
        model_metadata_1 = ModelMetadata(id=model_id_1, block=1)

        hotkey_2 = "test_hotkey2"
        model_id_2 = ModelId(
            namespace="test_model2",
            name="test_name2",
            commit="test_commit2",
            hash="test_hash2",
            secure_hash="test_secure_hash2",
            competition_id=1,
        )
        model_metadata_2 = ModelMetadata(id=model_id_2, block=2)

        self.model_tracker.on_miner_model_updated(hotkey_1, model_metadata_1)
        self.model_tracker.on_miner_model_updated(hotkey_2, model_metadata_2)

        hotkey_to_model_metadata = (
            self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
        )

        self.assertEqual(len(hotkey_to_model_metadata), 2)
        self.assertEqual(hotkey_to_model_metadata[hotkey_1], model_metadata_1)
        self.assertEqual(hotkey_to_model_metadata[hotkey_2], model_metadata_2)

    def test_on_hotkeys_updated_extra_ignored(self):
        hotkey = "test_hotkey"
        model_id = self._create_model_id()

        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.on_hotkeys_updated(set([hotkey, "extra_hotkey"]))

        self.assertEqual(len(self.model_tracker.miner_hotkey_to_model_metadata_dict), 1)

    def test_on_hotkeys_updated_missing_removed(self):
        hotkey = "test_hotkey"
        model_id = self._create_model_id()

        model_metadata = ModelMetadata(id=model_id, block=1)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.model_tracker.on_model_evaluated(
            hotkey,
            1,
            EvalResult(block=2, score=1, winning_model_block=1, winning_model_score=2),
        )
        self.model_tracker.on_hotkeys_updated(set(["extra_hotkey"]))

        self.assertEqual(len(self.model_tracker.miner_hotkey_to_model_metadata_dict), 0)
        self.assertEqual(len(self.model_tracker.miner_hotkey_to_eval_results), 0)

    def test_on_model_evaluated_cleared_on_metadata_update(self):
        hotkey = "test_hotkey"
        model_id = self._create_model_id()

        model_metadata = ModelMetadata(id=model_id, block=1)
        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)

        # Record a model evaluation.
        eval_result = EvalResult(
            block=2, score=1, winning_model_block=1, winning_model_score=2
        )
        self.model_tracker.on_model_evaluated(hotkey, 1, eval_result)
        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey(hotkey, 1),
            [eval_result],
        )

        # Call on_miner_model_updated with the same model_metadata. This shouldn't clear the eval results.
        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)
        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey(hotkey, 1),
            [eval_result],
        )

        # Now update the model metadata with a new metadata. This should clear the eval results.
        self.model_tracker.on_miner_model_updated(
            hotkey, ModelMetadata(id=model_id, block=2)
        )
        self.assertEqual(
            len(self.model_tracker.get_eval_results_for_miner_hotkey(hotkey, 1)), 0
        )

    def test_on_model_evaluated_multiple_miners(self):
        """Verifies that miner evaluations are stored per miner."""
        miner1 = "miner1"
        miner2 = "miner2"

        eval_result1 = EvalResult(
            block=1, score=1, winning_model_block=1, winning_model_score=2
        )
        eval_result2 = EvalResult(
            block=2, score=2, winning_model_block=1, winning_model_score=2
        )
        self.model_tracker.on_model_evaluated(miner1, 1, eval_result1)
        self.model_tracker.on_model_evaluated(miner2, 1, eval_result2)

        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey(miner1, 1),
            [eval_result1],
        )
        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey(miner2, 1),
            [eval_result2],
        )

    def test_on_model_evaluated_respects_competition_id(self):
        """Verifies that miner evaluations are stored per competition."""

        eval_result1 = EvalResult(
            block=1, score=1, winning_model_block=1, winning_model_score=2
        )
        eval_result2 = EvalResult(
            block=2, score=2, winning_model_block=1, winning_model_score=2
        )
        self.model_tracker.on_model_evaluated("miner", 1, eval_result1)
        self.model_tracker.on_model_evaluated("miner", 1, eval_result2)

        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey("miner", 1),
            [eval_result1, eval_result2],
        )
        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey("miner", 2), []
        )

    def test_eval_results_are_ordered_and_truncated(self):
        """Verifies that only 3 eval results are kept and that they're returned in block order."""

        miner1 = "miner1"
        eval_results = [
            EvalResult(
                block=i, score=10 - i, winning_model_block=1, winning_model_score=2
            )
            for i in range(50)
        ]
        random.shuffle(eval_results)

        # A bit unusual, but insert the evaluations in a random order.
        # Per the API spec, this should still result in the 3 most recent being kept.
        for eval_result in eval_results:
            self.model_tracker.on_model_evaluated(miner1, 1, eval_result)

        expected = sorted(eval_results, key=lambda er: er.block)[-3:]
        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey(miner1, 1), expected
        )

    def test_get_block_last_evaluated(self):
        hotkey = "hotkey"

        # 1. Check we get an empty result when no evals have been performed
        self.assertIsNone(self.model_tracker.get_block_last_evaluated(hotkey))

        # 2. Check we get the correct block when a single competition has been evaluated
        self.model_tracker.on_model_evaluated(
            hotkey,
            1,
            EvalResult(block=2, score=2, winning_model_block=1, winning_model_score=2),
        )
        self.assertEqual(self.model_tracker.get_block_last_evaluated(hotkey), 2)

        # 3. Now add another competition eval at an older block and check we still get the right result.
        self.model_tracker.on_model_evaluated(
            hotkey,
            2,
            EvalResult(block=1, score=1, winning_model_block=1, winning_model_score=2),
        )
        self.assertEqual(self.model_tracker.get_block_last_evaluated(hotkey), 2)

        # 4. Evaluate again on the new competition at a newer block.
        self.model_tracker.on_model_evaluated(
            hotkey,
            2,
            EvalResult(block=3, score=3, winning_model_block=1, winning_model_score=2),
        )
        self.assertEqual(self.model_tracker.get_block_last_evaluated(hotkey), 3)

        # 5. Finally, add another eval for the first competition.
        self.model_tracker.on_model_evaluated(
            hotkey,
            1,
            EvalResult(block=5, score=4, winning_model_block=1, winning_model_score=2),
        )
        self.assertEqual(self.model_tracker.get_block_last_evaluated(hotkey), 5)

    def test_clear_eval_results_missing_comp(self):
        """Verifies that clear_eval_results doesn't crash when the competition doesn't exist."""
        
        eval_result1 = EvalResult(
            block=1, score=1, winning_model_block=1, winning_model_score=2
        )
        eval_result2 = EvalResult(
            block=2, score=2, winning_model_block=1, winning_model_score=2
        )
        self.model_tracker.on_model_evaluated("miner", 1, eval_result1)
        self.model_tracker.on_model_evaluated("miner", 1, eval_result2)

        self.model_tracker.clear_eval_results(2)

        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey("miner", 1),
            [eval_result1, eval_result2],
        )

    def test_clear_eval_results(self):
        """Verifies that clear_eval_results only deletes results from the right competition."""

        eval_result1 = EvalResult(
            block=1, score=1, winning_model_block=1, winning_model_score=2
        )
        eval_result2 = EvalResult(
            block=2, score=2, winning_model_block=1, winning_model_score=2
        )
        self.model_tracker.on_model_evaluated("miner", 1, eval_result1)
        self.model_tracker.on_model_evaluated("miner", 1, eval_result1)
        self.model_tracker.on_model_evaluated("miner", 2, eval_result2)

        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey("miner", 1),
            [eval_result1, eval_result1],
        )
        self.model_tracker.clear_eval_results(1)
        self.assertEqual(
            self.model_tracker.get_eval_results_for_miner_hotkey("miner", 1), []
        )

if __name__ == "__main__":
    unittest.main()
