import os
import json

import torch

from safetensors.torch import load_file, save_file

from taoverse.model.pretrained import PreTrainedModel
from taoverse.model.tts.backbones.unett import UNetT
from taoverse.model.tts.backbones.cfm import CFM

from typing import List


class E2TTS(PreTrainedModel):
    """A wrapper for Microsoft's E2 TTS text-to-speech model.
    """

    def __init__(self,
                 cfm_model: CFM,
                 config: dict,
                 model_local_path: str,
                 ):

        super().__init__(model=cfm_model,
                         config=config,
                         model_local_path=model_local_path)

    @classmethod
    def from_pretrained(cls,
                        model_local_path: str,
                        **kwargs) -> "E2TTS":

        # Get the checkpoint file path
        # TODO: raise exception of file does not exist
        checkpoint_path = os.path.join(model_local_path,
                                       'model.safetensors')

        # Load the vocab
        vocab_char_map, vocab_size = cls.load_vocab(model_local_path)

        # Parse the config file
        model_config = cls.parse_config(model_local_path)

        # Get the necessary configs to load the model
        mel_spec_kwargs = model_config["mel_spectrogram"]
        backbone_kwargs = model_config["backbone"]
        backbone_kwargs["mel_dim"] = mel_spec_kwargs["n_mel_channels"]
        odeint_kwargs = model_config["odeint"]

        # Init the backbone
        backbone = UNetT(**backbone_kwargs,
                        text_num_embeds=vocab_size)

        # Init the CFM model
        cfm_model = CFM(
            transformer=backbone,
            mel_spec_kwargs=mel_spec_kwargs,
            odeint_kwargs=odeint_kwargs,
            vocab_char_map=vocab_char_map,
        )

        # Load checkpoint
        checkpoint = load_file(checkpoint_path)

        cfm_model.load_state_dict(checkpoint)

        # Initial a model instance
        e2tts = cls(cfm_model=cfm_model,
                    config=model_config,
                    model_local_path=model_local_path)


        return e2tts

    @classmethod
    def load_vocab(cls, model_local_path: str):
        """ Load and return the vocab-related objectis
        """

        # TODO: raise exception of file does not exist
        vocab_path = os.path.join(model_local_path,
                                  'vocab.txt')

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                # -1 to remove the newline character
                vocab_char_map[char[:-1]] = i
                vocab_size = len(vocab_char_map)

        return vocab_char_map, vocab_size


    def sample(self,
               ref_audio: torch.tensor,
               text: List[str],
               gen_duration: int
               ):


        generated_wave, _ =  self.model.sample(
            cond=ref_audio,
            text=text,
            duration=gen_duration,
            steps=self.config['sampling']['nfe_step'],
            cfg_strength=self.config['sampling']['cfg_strength'],
            sway_sampling_coef=self.config['sampling']['sway_sampling_coef']
        )


        return generated_wave
