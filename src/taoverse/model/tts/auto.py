import json
import os

from huggingface_hub import HfApi

class AutoModelForTTS:

    @classmethod
    def get_model_class(cls, model_local_path: str):
        """
        Read the config.json file of the model and return the appropriate model class
        """

        config_path = os.path.join(model_local_path, 'config.json')

        with open(config_path) as f:
            model_config = json.load(f)

        # Get the architecture name from the config, and convert to lowercase
        arch_list = model_config.get("architectures", [])

        if arch_list:
            arch = arch_list[0].lower()
        else:
            raise ValueError(f"No architecture specified in the config file for model {model_local_path}.")

        match arch:
            case "e2tts":
                from taoverse.model.tts.e2tts import E2TTS
                return E2TTS
            case _:
                raise ValueError(
                    f"Failed to load model: {model_local_path}. "
                    f"Unknown architecture: {arch} found in config.json"
                )

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        local_files_only: bool=False,
                        revision: str=None,
                        cache_dir: str=None,
                        token: str=None,
                        **kwargs
                        ):
        """
        Load the pretrained model using the appropriate model class.
        """

        # Get the model's local path or create one it if the model is remote, after
        # downloading from HuggingFace Hub
        model_local_path = cls.get_model_local_path(pretrained_model_name_or_path,
                                                    local_files_only,
                                                    revision,
                                                    cache_dir,
                                                    token
                                                    )

        model_cls = cls.get_model_class(model_local_path)

        # Load the pretrained model
        pretrained_model = model_cls.from_pretrained(model_local_path)

        return pretrained_model

    @classmethod
    def get_model_local_path(cls,
                             pretrained_model_name_or_path: str,
                             local_files_only: bool=False,
                             revision: str=None,
                             cache_dir: str=None,
                             token: str=None,
                             ) -> str:

        # If the local config file exists, then this might be a local model
        # load request
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")

        if os.path.isfile(config_path):
            return pretrained_model_name_or_path

        # Assum that we need to get the model for Hugging Face since no
        # local model path is found
        # Note: do this only if local_files_only is False

        elif local_files_only:
            raise EnvironmentError(
                    f"Local model files not found at {pretrained_model_name_or_path} but `local_files_only` is set to True, ")

        else:


            # Download the model folder from huggingface
            # and return the download path
            model_local_path = cls.download_from_hub(repo_id=pretrained_model_name_or_path,
                                                     revision=revision,
                                                     cache_dir=cache_dir,
                                                     token=token)

            return model_local_path

    @classmethod
    def download_from_hub(cls,
                          repo_id: str,
                          revision: str=None,
                          cache_dir: str=None,
                          token: str=None,
                          ) -> str:
        """Download a model from HuggingFace given its repo ID
        """
        api = HfApi()
        model_local_path = api.snapshot_download(repo_id=repo_id,
                                                 revision=revision,
                                                 cache_dir=cache_dir,
                                                 token=token,
                                                 )

        return model_local_path
