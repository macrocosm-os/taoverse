import os
import tempfile
from dataclasses import replace
from typing import Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModelForCausalLM

from taoverse.model.competition.data import ModelConstraints
from taoverse.model.data import Model, ModelId
from taoverse.model.model_updater import MinerMisconfiguredError
from taoverse.model.storage.disk import utils
from taoverse.model.storage.remote_model_store import RemoteModelStore


class HuggingFaceModelStore(RemoteModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    @classmethod
    def assert_access_token_exists(cls) -> str:
        """Asserts that the access token exists."""
        if not os.getenv("HF_ACCESS_TOKEN"):
            raise ValueError("No Hugging Face access token found to write to the hub.")
        return os.getenv("HF_ACCESS_TOKEN")

    @classmethod
    def get_access_token_if_exists(cls) -> Optional[str]:
        """Returns the access token if it exists."""
        return os.getenv("HF_ACCESS_TOKEN")

    async def upload_model(
        self, model: Model, model_constraints: ModelConstraints
    ) -> ModelId:
        """Uploads a trained model to Hugging Face."""
        token = HuggingFaceModelStore.assert_access_token_exists()

        commit_info = model.pt_model.push_to_hub(
            repo_id=model.id.namespace + "/" + model.id.name,
            token=token,
            safe_serialization=True,
            private=True,
        )

        model_id_with_commit = replace(model.id, commit=commit_info.oid)

        # To make sure we get the same hash as validators, we need to redownload it at a
        # local tmp directory after which it can be deleted.
        with tempfile.TemporaryDirectory() as temp_dir:
            model_with_hash = await self.download_model(
                model_id_with_commit, temp_dir, model_constraints
            )
            # Return a ModelId with both the correct commit and hash.
            return model_with_hash.id

    async def download_model(
        self,
        model_id: ModelId,
        local_path: str,
        model_constraints: ModelConstraints,
    ) -> Model:
        """Retrieves a trained model from Hugging Face."""
        if not model_id.commit:
            raise ValueError("No Hugging Face commit id found to read from the hub.")

        repo_id = model_id.namespace + "/" + model_id.name
        token = HuggingFaceModelStore.get_access_token_if_exists()

        # Check ModelInfo for the size of model.safetensors file before downloading.
        api = HfApi()
        try:
            model_info = api.model_info(
                repo_id=repo_id,
                revision=model_id.commit,
                timeout=10,
                files_metadata=True,
                token=token,
            )
        except RepositoryNotFoundError:
            raise MinerMisconfiguredError(
                hotkey="N/A",
                message=f"HuggingFace repository {repo_id} with revision {model_id.commit} was not found on the hub.",
            )

        size = sum(repo_file.size for repo_file in model_info.siblings)
        if size > model_constraints.max_bytes:
            raise MinerMisconfiguredError(
                hotkey="N/A",
                message=f"Hugging Face repo over maximum size limit. Size {size}. Limit {model_constraints.max_bytes}.",
            )

        # Transformers library can pick up a model based on the hugging face path (username/model) + rev.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=repo_id,
                revision=model_id.commit,
                cache_dir=local_path,
                use_safetensors=True,
                token=token,
                **model_constraints.kwargs,
            )
        except ValueError as e:
            # This is treated as a MinerMisconfiguredError. Since the error in this case,
            # and after the above checks, most probably comes from `kwargs` misconfigured for
            # the type of model being loaded.
            # For example, attempting to use FlashAttention 2.0 with GPT2 models.
            # The latter does not support that. This is an error known for SN9 when trying
            # to load 772M models into the default 7B when migrating to multi-competition support.
            raise MinerMisconfiguredError(
                hotkey="N/A",
                message=f"Model {repo_id}/{model_id.commit} could not be loaded with kwargs {model_constraints.kwargs}, Here is the error trace:",
            ) from e

        # Get the directory the model was stored to.
        model_dir = utils.get_hf_download_path(local_path, model_id)

        # Realize all symlinks in that directory since Transformers library does not support avoiding symlinks.
        utils.realize_symlinks_in_directory(model_dir)

        # Compute the hash of the downloaded model.
        model_hash = utils.get_hash_of_directory(model_dir)
        model_id_with_hash = replace(model_id, hash=model_hash)

        return Model(id=model_id_with_hash, pt_model=model)
