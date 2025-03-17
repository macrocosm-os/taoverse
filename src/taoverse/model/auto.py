import os
import json

from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM

from taoverse.model.tts.auto import AutoModelForTTS

# Define a dict that maps a given model class
# to its auto model class
# TODO: Move this to a config file
_AUTO_MAPPINGS = {
    'e2tts': AutoModelForTTS
}

class AutoModel:

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        local_files_only: bool=False,
                        revision: str=None,
                        cache_dir: str=None,
                        token: str=None,
                        use_safetensors: bool=True,
                        **kwargs,
        ):

        # Get the model config path
        config_path = cls.get_config_path(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                          revision=revision,
                                          local_files_only=local_files_only,
                                          cache_dir=cache_dir,
                                          token=token)

        with open(config_path) as f:
            model_config = json.load(f)

        # Get the architecture name from the config, and convert to lowercase
        arch_list = model_config.get("architectures", [])

        if arch_list:
            arch = arch_list[0].lower()
        else:
            raise ValueError(f"No architecture specified in the config file for model {pretrained_model_name_or_path}.")


        # Get the Auto class. Default to HuggingFace's AutoModelForCausalLM
        auto_class = _AUTO_MAPPINGS.get(arch, AutoModelForCausalLM)

        return auto_class.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                          revision=revision,
                                          local_files_only=local_files_only,
                                          cache_dir=cache_dir,
                                          token=token,
                                          use_safetensors=use_safetensors,
                                          **kwargs)

    @classmethod
    def get_config_path(cls,
                        pretrained_model_name_or_path: str,
                        local_files_only: bool=False,
                        revision: str=None,
                        cache_dir: str=None,
                        token: str=None) -> str:


        if local_files_only:
            config_path = os.path.join(pretrained_model_name_or_path, 'config.json')

        else:
            # Download the config file
            # TODO: Test using this class instead of pretraining models.factory.py
            api = HfApi()
            config_path = api.hf_hub_download(repo_id=pretrained_model_name_or_path,
                                              revision=revision,
                                              cache_dir=cache_dir,
                                              token=token,
                                              filename="config.json")


        return config_path
