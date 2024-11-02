import dataclasses

@dataclasses.dataclass
class EvalTask:
    """Represents a task to evaluate a model on."""

    # Friendly name for the task.
    name: str

    # The identifier of the evaluation method to use.
    method_id: int
    
    # The identifier of the dataset to evaluate on.
    dataset_id: int
    
    # Additional keyword arguments to pass to the dataset loader.
    dataset_kwargs: dict = dataclasses.field(default_factory=dict)

    # The identifier of the normalization method to use.
    normalization_id: int

    # Additional keyword arguments to pass to the normalization method.
    normalization_kwargs: dict = dataclasses.field(default_factory=dict)

    # Weight to apply to the normalized score to provide relative weight against other EvalTasks.
    weight: float = 1.0

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Weight must be positive.")
