from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any, List, Optional
from enum import Enum
from GANDLF.models.modelBase import ModelBase
from typing import Union


class Version(BaseModel):
    minimum: str
    maximum: str


class Model(BaseModel):
    dimension: int
    base_filters: int
    architecture: str
    norm_type: str
    final_layer: str
    class_list: list[Union[int, str]]
    ignore_label_validation: Union[int, None]
    amp: bool
    print_summary: bool
    type: str
    data_type: str
    save_at_every_epoch: bool
    num_channels: Optional[int] = None


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Version
    model: Model
    modality: str
    scheduler: dict
    learning_rate: float
    weighted_loss: bool
    verbose: bool
    q_verbose: bool
    medcam_enabled: bool
    save_training: bool
    save_output: bool
    in_memory: bool
    pin_memory_dataloader: bool
    scaling_factor: Union[float, int]
    q_max_length: int
    q_samples_per_volume: int
    q_num_workers: int
    num_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    clip_grad: Union[None, float]
    track_memory_usage: bool
    memory_save_mode: bool
    print_rgb_label_warning: bool
    data_postprocessing: Dict  # TODO: maybe is better to create a class
    data_preprocessing: Dict  # TODO: maybe is better to create a class
    grid_aggregator_overlap: str
    determinism: bool
    previous_parameters: None
    metrics: Union[List, dict]
    patience: int
    parallel_compute_command: Union[str, bool, None]
    loss_function: Union[str, Dict]
    data_augmentation: dict  # TODO: maybe is better to create a class
    nested_training: dict  # TODO: maybe is better to create a class
    optimizer: Union[dict, str]
    patch_sampler: Union[dict, str]
    patch_size: Union[List[int], int]
    clip_mode: Union[str, None]
    inference_mechanism: dict
    data_postprocessing_after_reverse_one_hot_encoding: dict
    enable_padding: Optional[Union[dict, bool]] = None
    headers: Optional[dict] = None
    output_dir: Optional[str] = ""
    problem_type: Optional[str] = None
    differential_privacy: Optional[dict] = None
    # opt: Optional[Union[dict, str]] = {}  # TODO find a better way
