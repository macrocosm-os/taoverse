import dataclasses

from taoverse.model.eval.normalization import NormalizationId

@dataclasses.dataclass
class EvalTask:
    """Represents a task to evaluate a model on."""

    # Friendly name for the task.
    name: str

    # The identifier of the evaluation method to use.
    method_id: int
    
    # The identifier of the dataset to evaluate on.
    dataset_id: int

    # The identifier of the normalization method to use.
    normalization_id: NormalizationId = NormalizationId.NONE

    # Additional keyword arguments to pass to the normalization method.
    normalization_kwargs: dict = dataclasses.field(default_factory=dict)
    
    # Additional keyword arguments to pass to the dataset loader.
    dataset_kwargs: dict = dataclasses.field(default_factory=dict)

    # Weight to apply to the normalized score to provide relative weight against other EvalTasks.
    weight: float = 1.0

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Weight must be positive.")

        match self.normalization_id:
            case NormalizationId.NONE:
                if self.normalization_kwargs:
                    raise ValueError(
                        "Normalization kwargs should not be provided for NONE normalization."
                    )
            case NormalizationId.INVERSE_EXPONENTIAL:
                if "ceiling" not in self.normalization_kwargs:
                    raise ValueError(
                        "Normalization kwargs must contain a 'ceiling' value."
                    )