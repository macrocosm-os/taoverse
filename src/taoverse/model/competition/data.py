from dataclasses import dataclass, field
import dataclasses
from typing import Any, List, Optional, Type

from transformers import PreTrainedModel

from taoverse.model.competition.epsilon import EpsilonFunc
from taoverse.model.eval.task import EvalTask


@dataclass
class NormValidationConstraints:
    """Defines the validation constriants to be checked based on model norms."""

    norm_eps_soft: int
    norm_eps_soft_percent_threshold: float
    norm_eps_hard: int


@dataclass
class ModelConstraints:
    """Defines the constraints for models submitted to a specific competition."""

    # The maximum parameter size allowed for models
    max_model_parameter_size: int

    # Architecture class of model
    allowed_architectures: List[Type[PreTrainedModel]]

    # The model's sequence length.
    sequence_length: int

    # The Pretrained tokenizer to use.
    tokenizer: str

    # Block delay before evaluating uploaded models. Based on look-back period for eval data collection.
    eval_block_delay: int

    # The function to compute epsilon as a function of the model's submitted block and the current block.
    epsilon_func: EpsilonFunc = field(compare=False)

    # The maximum size (in bytes) the model is allowed to be.
    max_bytes: int

    # Any additional arguments to pass to from_pretrained
    kwargs: Any = field(default_factory=dict)

    # The minimum parameter size allowed for models
    min_model_parameter_size: int = 0

    # Norm validation values.
    norm_validation_constraints: Optional[NormValidationConstraints] = None


@dataclass
class Competition:
    """Defines a competition."""

    # Unique ID for this competition.
    # Recommend making an IntEnum for use in the subnet codebase.
    id: int

    # All restrictions on models allowed in this competition.
    constraints: ModelConstraints

    # Percentage of emissions dedicated to this competition.
    reward_percentage: float

    # The set of tasks used to evaluate models in this competition.
    eval_tasks: List[EvalTask] = dataclasses.field(default_factory=list)
