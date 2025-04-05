from pydantic import BaseModel, ConfigDict, Field, AliasChoices, model_validator
from typing_extensions import Any, Literal, Self


class ThresholdConfig(BaseModel):
    min: int = Field()
    max: int = Field()


class ClipConfig(BaseModel):
    min: int = Field()
    max: int = Field()


class RescaleConfig(BaseModel):
    in_min_max: list[float] = Field(default=[15, 125])
    out_min_max: list[float] = Field(default=[0, 1])
    percentiles: list[float] = Field(default=[5, 95])


class HistogramMatchingConfig(BaseModel):
    num_hist_level: int = Field(default=1024)
    num_match_points: int = Field(default=16)
    target: Any = Field(default=None)


class ResampleMinConfig(BaseModel):
    resolution: list[float] = Field(default=None)


class ResampleConfig(BaseModel):
    resolution: list[float] = Field(default=None)


class StainNormalizationConfig(BaseModel):
    target: Any = Field()
    extractor: Literal["vahadane", "ruifrok", "macenko"] = Field(default="ruifrok")


class PreProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)
    to_canonical: Any = Field(default=None)
    threshold: ThresholdConfig = Field(default=None)
    clip: ClipConfig = Field(default=None)
    clamp: ClipConfig = Field(default=None)
    crop_external_zero_planes: Any = Field(default=None)
    crop: list[int] = Field(default=None)
    centercrop: list[int] = Field(default=None)
    normalize_by_val: Any = Field(default=None)
    normalize_imagenet: Any = Field(default=None)
    normalize_standardize: Any = Field(default=None)
    normalize_div_by_255: Any = Field(default=None)
    normalize: Any = Field(default=None)
    normalize_nonZero: Any = Field(
        default=None,
        validation_alias=AliasChoices("normalize_nonZero", "normalize_nonzero"),
    )
    normalize_nonZero_masked: Any = Field(
        default=None,
        validation_alias=AliasChoices(
            "normalize_nonZero_masked", "normalize_nonzero_masked"
        ),
    )
    rescale: RescaleConfig = Field(default=None)
    rgba2rgb: Any = Field(
        default=None,
        validation_alias=AliasChoices("rgba2rgb", "rgbatorgb", "rgba_to_rgb"),
    )
    rgb2rgba: Any = Field(
        default=None,
        validation_alias=AliasChoices("rgb2rgba", "rgbtorgba", "rgb_to_rgba"),
    )
    histogram_matching: HistogramMatchingConfig = Field(default=None)
    histogram_equalization: HistogramMatchingConfig = Field(default=None)
    adaptive_histogram_equalization: Any = Field(default=None)
    resample: ResampleConfig = Field(default=None)
    resize_image: list[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "resize_image", "resize", "resize_image", "resize_images"
        ),
    )
    resize_patch: list[int] = Field(default=None)
    stain_normalization: StainNormalizationConfig = Field(default=None)
    resample_min: ResampleMinConfig = Field(
        default=None, validation_alias=AliasChoices("resample_min", "resample_minimum")
    )

    @model_validator(mode="after")
    def pre_processing_validate(self) -> Self:
        if self.adaptive_histogram_equalization is not None:
            self.histogram_matching = HistogramMatchingConfig(target="adaptive")
            self.adaptive_histogram_equalization = None
        return self
