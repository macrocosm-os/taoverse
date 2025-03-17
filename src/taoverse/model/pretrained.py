import os
import json

import torch.nn as nn
from huggingface_hub import HfApi

class PreTrainedModel:
    """This is a base class that provide some common functionalities
    to Taover models such as pushing to HF hub, etc.
    """

    def __init__(self,
                 model: nn.Module,
                 config: dict,
                 model_local_path: str):

        self.model = model
        self.config = config
        self.model_local_path = model_local_path

    def push_to_hub(self,
                    repo_id:str,
                    token:str,
                    **kwargs):
        """Pushes the 'already existing' model's local repo to HF.

        TODO: We should also support pushing a model that is not yet
              locally saved.
        """

        api = HfApi()

        # Create repo if not existing yet
        api.create_repo(repo_id, exist_ok=True)

        # Upload and get commit info
        commit_info = api.upload_folder(repo_id=repo_id,
                                        token=token,
                                        folder_path=self.model_local_path)

        return commit_info

    @classmethod
    def parse_config(cls, model_local_path: str):
        """Parse the model config file
        """

        config_path = os.path.join(model_local_path, 'config.json')

        with open(config_path) as f:
            model_config = json.load(f)


        return model_config


    def parameters(self, recurse: bool=True):

        return self.model.parameters(recurse=recurse)

    def to(self, device, *args, **kwargs):

        # Move the model to the specified device
        self.model = self.model.to(device, *args, **kwargs)
        return self

    def eval(self, *args, **kwargs):

        self.model.eval(*args, **kwargs)
        return self
