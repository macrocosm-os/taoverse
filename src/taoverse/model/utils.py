import base64
import hashlib
from typing import Tuple

from taoverse.model.data import ModelId, ModelMetadata


def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Verifies a Hugging Face repo id is valid and returns it split into namespace and name.

    Raises:
        ValueError: If the repo id is invalid.
    """

    if not repo_id:
        raise ValueError("Hugging Face repo id cannot be empty.")

    if not 3 < len(repo_id) <= ModelId.MAX_REPO_ID_LENGTH:
        raise ValueError(
            f"Hugging Face repo id must be between 3 and {ModelId.MAX_REPO_ID_LENGTH} characters."
        )

    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Hugging Face repo id must be in the format <org or user name>/<repo_name>."
        )

    return parts[0], parts[1]


def get_hf_repo_name(model_metadata: ModelMetadata) -> str:
    """Returns the Hugging Face repo name for the provided model metadata."""
    return f"{model_metadata.id.namespace}/{model_metadata.id.name}"


def get_hf_url(model_metadata: ModelMetadata) -> str:
    """Returns the URL to the Hugging Face repo for the provided model metadata."""
    return f"https://huggingface.co/{get_hf_repo_name(model_metadata)}/tree/{model_metadata.id.commit}"


def get_hash_of_two_strings(string1: str, string2: str) -> str:
    """Hashes two strings together and returns the result."""

    string_hash = hashlib.sha256((string1 + string2).encode())

    return base64.b64encode(string_hash.digest()).decode("utf-8")
