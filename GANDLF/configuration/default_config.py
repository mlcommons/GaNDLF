from pydantic import BaseModel, Field, AfterValidator
from typing import Dict
from typing_extensions import Literal, Optional, Annotated

from GANDLF.configuration.validators import validate_postprocessing

GRID_AGGREGATOR_OVERLAP_OPTIONS = Literal["crop", "average", "hann"]


class DefaultParameters(BaseModel):
    weighted_loss: bool = Field(
        default=False, description="Whether weighted loss is to be used or not."
    )
    verbose: bool = Field(default=False, description="General application verbosity.")
    q_verbose: bool = Field(default=False, description="Queue construction verbosity.")
    medcam_enabled: bool = Field(
        default=False, description="Enable interpretability via medcam."
    )
    save_training: bool = Field(
        default=False, description="Save outputs during training."
    )
    save_output: bool = Field(
        default=False, description="Save outputs during validation/testing."
    )
    in_memory: bool = Field(default=False, description="Pin data to CPU memory.")
    pin_memory_dataloader: bool = Field(
        default=False, description="Pin data to GPU memory."
    )
    scaling_factor: int = Field(
        default=1, description="Scaling factor for regression problems."
    )
    q_max_length: int = Field(default=100, description="The max length of the queue.")
    q_samples_per_volume: int = Field(
        default=10, description="Number of samples per volume."
    )
    q_num_workers: int = Field(
        default=0, description="Number of worker threads to use."
    )
    num_epochs: int = Field(default=100, description="Total number of epochs to train.")
    patience: int = Field(
        default=100, description="Number of epochs to wait for performance improvement."
    )
    batch_size: int = Field(default=1, description="Default batch size for training.")
    learning_rate: float = Field(default=0.001, description="Default learning rate.")
    clip_grad: Optional[float] = Field(
        default=None, description="Gradient clipping value."
    )
    track_memory_usage: bool = Field(
        default=False, description="Enable memory usage tracking."
    )
    memory_save_mode: bool = Field(
        default=False,
        description="Enable memory-saving mode. If enabled, resize/resample will save files to disk.",
    )
    print_rgb_label_warning: bool = Field(
        default=True, description="Print a warning for RGB labels."
    )
    data_postprocessing: Annotated[
        dict,
        Field(description="Default data postprocessing configuration.", default={}),
        AfterValidator(validate_postprocessing),
    ]

    grid_aggregator_overlap: GRID_AGGREGATOR_OVERLAP_OPTIONS = Field(
        default="crop", description="Default grid aggregator overlap strategy."
    )
    determinism: bool = Field(
        default=False, description="Enable deterministic computation."
    )
    previous_parameters: Optional[Dict] = Field(
        default=None,
        description="Previous parameters to be used for resuming training and performing sanity checks.",
    )
