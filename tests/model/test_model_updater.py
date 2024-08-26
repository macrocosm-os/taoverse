import asyncio
import unittest
from typing import Dict, List, Tuple

import torch
from transformers import (
    BartForCausalLM,
    FalconForCausalLM,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    PhiForCausalLM,
)

from taoverse.model import utils
from taoverse.model.competition.data import (
    Competition,
    ModelConstraints,
    NormValidationConstraints,
)
from taoverse.model.competition.epsilon import FixedEpsilon
from taoverse.model.data import Model, ModelId, ModelMetadata
from taoverse.model.model_tracker import ModelTracker
from taoverse.model.model_updater import MinerMisconfiguredError, ModelUpdater
from taoverse.model.storage.disk.disk_model_store import DiskModelStore
from tests.model.storage.fake_model_metadata_store import FakeModelMetadataStore
from tests.model.storage.fake_remote_model_store import FakeRemoteModelStore
from tests.utils import get_test_model


class TestModelUpdater(unittest.TestCase):
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

    def setUp(self):
        self.model_tracker = ModelTracker()
        self.local_store = DiskModelStore("test-models")
        self.remote_store = FakeRemoteModelStore()
        self.metadata_store = FakeModelMetadataStore()
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )
        self.tiny_model = get_test_model()

    def tearDown(self):
        self.local_store.delete_unreferenced_models(dict(), 0)

    def test_get_metadata(self):
        hotkey = "test_hotkey"
        model_hash = "TestHash1"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=model_hash,
            secure_hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="TestCommit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )

        metadata = asyncio.run(self.model_updater._get_metadata(hotkey))

        self.assertEqual(metadata.id, model_id)
        self.assertIsNotNone(metadata.block)

    def test_sync_model_bad_metadata(self):
        hotkey = "test_hotkey"
        model_hash = "TestHash1"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=model_hash,
            secure_hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="TestCommit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        # Setup the metadata with a commit that doesn't exist in the remote store.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )

        # FakeRemoteModelStore raises a KeyError but HuggingFace may raise other exceptions.
        with self.assertRaises(Exception):
            asyncio.run(
                self.model_updater.sync_model(
                    hotkey=hotkey,
                    curr_block=100_000,
                    schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
                )
            )

    def test_sync_model_same_metadata(self):
        hotkey = "test_hotkey"
        model_hash = "TestHash1"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=model_hash,
            secure_hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="TestCommit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        pt_model = self.tiny_model

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata, local, and model_tracker to match.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        self.local_store.store_model(hotkey, model)

        self.model_tracker.on_miner_model_updated(hotkey, model_metadata)

        asyncio.run(
            self.model_updater.sync_model(
                hotkey=hotkey,
                curr_block=100_000,
                schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
            )
        )

        # Tracker information did not change.
        self.assertEqual(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey),
            model_metadata,
        )

    def test_sync_model_new_metadata(self):
        hotkey = "test_hotkey"
        model_hash = "TestHash1"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=model_hash,
            secure_hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="TestCommit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        pt_model = self.tiny_model

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store but not local or the model_tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        asyncio.run(
            self.remote_store.upload_model(
                model,
                self.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(self.COMPETITION_ID),
            )
        )

        self.assertIsNone(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        )

        # Our local store raises an exception from the Transformers.from_pretrained method if not found.
        with self.assertRaises(Exception):
            self.local_store.retrieve_model(hotkey, model_id, kwargs={})

        asyncio.run(
            self.model_updater.sync_model(
                hotkey=hotkey,
                curr_block=100_000,
                schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
            )
        )

        self.assertEqual(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey),
            model_metadata,
        )
        self.assertEqual(
            str(self.local_store.retrieve_model(hotkey, model_id, kwargs={})),
            str(model),
        )

    def test_sync_model_new_metadata_under_block_delay(self):
        hotkey = "test_hotkey"
        model_hash = "TestHash1"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=model_hash,
            secure_hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="TestCommit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        pt_model = self.tiny_model

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store but not local or the model_tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        asyncio.run(
            self.remote_store.upload_model(
                model,
                self.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(self.COMPETITION_ID),
            )
        )

        self.assertIsNone(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        )

        updated = asyncio.run(
            self.model_updater.sync_model(
                hotkey=hotkey,
                curr_block=1,
                schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
            )
        )

        # Tracker information did not change.
        self.assertFalse(updated)
        self.assertIsNone(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        )

    def test_sync_model_bad_hash(self):
        hotkey = "test_hotkey"
        model_hash = "TestHash1"
        model_id_chain = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=model_hash,
            secure_hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="TestCommit",
        )

        model_metadata = ModelMetadata(id=model_id_chain, block=1)
        bad_hash = "BadHash"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=bad_hash,
            secure_hash=utils.get_hash_of_two_strings(bad_hash, hotkey),
            commit="TestCommit",
        )

        pt_model = self.tiny_model

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store and but not local or the model tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        self.remote_store.inject_mismatched_model(model_id_chain, model)

        # Assert we fail due to the hash mismatch between the model in remote store and the metadata on chain.
        with self.assertRaises(MinerMisconfiguredError) as context:
            asyncio.run(
                self.model_updater.sync_model(
                    hotkey=hotkey,
                    curr_block=100_000,
                    schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
                )
            )

        self.assertIn("Hash", str(context.exception))

        # Also assert that the model tracker is tracking the model even after the exception above.
        self.assertEqual(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey),
            model_metadata,
        )

    # TODO: Create separate tests for size, architecture, and eps.
    def test_sync_model_wrong_parameters(self):
        hotkey = "test_hotkey"
        model_hash = "TestHash1"
        model_id = ModelId(
            namespace="TestPath",
            name="TestModel",
            competition_id=self.COMPETITION_ID,
            hash=model_hash,
            secure_hash=utils.get_hash_of_two_strings(model_hash, hotkey),
            commit="TestCommit",
        )
        model_metadata = ModelMetadata(id=model_id, block=1)

        config = GPT2Config(
            n_head=10,
            n_layer=12,
            n_embd=760,
        )
        pt_model = GPT2LMHeadModel(config)

        model = Model(id=model_id, pt_model=pt_model)

        # Setup the metadata and remote store but not local or the model_tracker.
        asyncio.run(
            self.metadata_store.store_model_metadata_exact(hotkey, model_metadata)
        )
        asyncio.run(
            self.remote_store.upload_model(
                model,
                self.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(self.COMPETITION_ID),
            ),
        )

        # Assert we fail due to not meeting the competition parameters.
        with self.assertRaises(MinerMisconfiguredError) as context:
            asyncio.run(
                self.model_updater.sync_model(
                    hotkey=hotkey,
                    curr_block=100_000,
                    schedule_by_block=self.COMPETITION_SCHEDULE_BY_BLOCK,
                )
            )

        self.assertIn("does not satisfy parameters", str(context.exception))

        # Also assert that the model tracker is tracking the model even after the exception above.
        self.assertEqual(
            self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey),
            model_metadata,
        )

    # TODO: Create test for valid competition at too early of a block once added.


if __name__ == "__main__":
    unittest.main()
