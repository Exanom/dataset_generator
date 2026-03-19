from typing import Annotated, Literal, Optional
from typing_extensions import Self
from pydantic import BaseModel, Field, field_validator, model_validator
import math
import random


# drift defs
class DriftDef(BaseModel):
    center: int = Field(gt=0)
    window: int = Field(gt=0, default=1)

    @model_validator(mode="after")
    def min_window_check(self) -> Self:
        if self.center - math.ceil(self.window / 2) < 0:
            raise ValueError(
                f"Drift with the window of size {self.window} and the center point of {self.center} would start before the first sample."
            )
        return self


# Feature defs
class FeatureDist(BaseModel):
    type: Literal["continuous"]
    dist_mean: float
    dist_std: float = Field(gt=0)
    min_val: float
    max_val: float

    @model_validator(mode="after")
    def range_check(self) -> Self:
        if self.min_val >= self.dist_mean or self.max_val <= self.dist_mean:
            raise ValueError(
                f"Minimum value and maximum value for a feature must  be lesser than and greater than the distribution mean respectively. Found inconsistency for mean {self.dist_mean} and range ({self.min_val} ; {self.max_val})."
            )
        return self


class FeatureDistLiteral(BaseModel):
    type: Literal["categorical"]
    literals: list[str]
    probabilities: list[float]  # probability for each literal to be assigned

    @field_validator("probabilities", mode="after")
    @classmethod
    def probabilities_check(cls, vals: list[float]) -> list[float]:
        prob_sum = 0
        for val in vals:
            if val < 0 or val > 1:
                raise ValueError(
                    "Probabilities for categorical features must be between 0 and 1."
                )
            prob_sum += val
        if not math.isclose(prob_sum, 1, rel_tol=1e-2):
            raise ValueError("Probabilities for categorical features must sum up to 1.")

        return vals

    @model_validator(mode="after")
    def length_check(self) -> Self:
        if len(self.literals) != len(self.probabilities):
            raise ValueError(
                "Categorical features require a probability set for each literal."
            )
        return self


DistType = Annotated[FeatureDist | FeatureDistLiteral, Field(discriminator="type")]


class FeatureDrift(BaseModel):
    distributions: list[DistType] = Field(min_length=1)
    drift_defs: list[DriftDef]

    @model_validator(mode="after")
    def drift_number_check(self) -> Self:
        if len(self.drift_defs) != (len(self.distributions) - 1):
            raise ValueError(
                f"Each change in data distribution must have a corresponding drift definition. Found {len(self.distributions)} distributions, but {len(self.drift_defs)} drift definitions."
            )
        return self

    @model_validator(mode="after")
    def drift_strictly_rising_check(self) -> Self:
        last = 0
        for drift in self.drift_defs:
            if drift.center <= last:
                raise ValueError(
                    f"Consecuting drift center points must be strictly rising. Found {drift.center} set after {last}."
                )
            last = drift.center
        return self

    @model_validator(mode="after")
    def drift_overlap_check(self) -> Self:
        last_end = 0
        for drift in self.drift_defs:
            start = drift.center - math.ceil(drift.window / 2)  # conservative, round up
            end = drift.center + math.ceil(drift.window / 2)  # conservative, round up
            if start < last_end:
                raise ValueError(
                    f"Data drift occurences for a given feature cannot overlap. Drift occuring between {start} and {end}, centered at {drift.center}, would overlap with previous drift that finishes at {last_end}."
                )
            last_end = end
        return self


class Feature(BaseModel):
    name: str
    type: Literal["str", "int", "float"]
    data_dist: FeatureDrift

    @model_validator(mode="after")
    def feature_type_check(self) -> Self:
        for dist in self.data_dist.distributions:
            if self.type == "str":
                if dist.type != "categorical":
                    raise ValueError(f"Type mismatch for feature {self.name}.")
            else:
                if dist.type != "continuous":
                    raise ValueError(f"Type mismatch for feature {self.name}.")
        return self


# classification function defs


class ClassFunc(BaseModel):
    functions: list[str] = Field(min_length=1)  # will be securely passed to simpleeval
    drift_defs: list[DriftDef]

    @model_validator(mode="after")
    def drift_number_check(self) -> Self:
        if len(self.drift_defs) != (len(self.functions) - 1):
            raise ValueError(
                f"Each change in classification function must have a corresponding drift definition. Found {len(self.functions)} classification functions, but {len(self.drift_defs)} drift definitions."
            )
        return self

    @model_validator(mode="after")
    def drift_strictly_rising_check(self) -> Self:
        last = 0
        for drift in self.drift_defs:
            if drift.center <= last:
                raise ValueError(
                    f"Consecuting drift center points must be strictly rising. Found {drift.center} set after {last}."
                )
            last = drift.center
        return self

    @model_validator(mode="after")
    def drift_overlap_check(self) -> Self:
        last_end = 0
        for drift in self.drift_defs:
            start = drift.center - math.ceil(drift.window / 2)  # conservative, round up
            end = drift.center + math.ceil(drift.window / 2)  # conservative, round up
            if start < last_end:
                raise ValueError(
                    f"Concept drift occurences cannot overlap. Drift occuring between {start} and {end}, centered at {drift.center}, would overlap with previous drift that finishes at {last_end}."
                )
            last_end = end
        return self


# main def
class DatasetDef(BaseModel):
    name: str
    features: list[Feature] = Field(min_length=1)
    class_func: ClassFunc
    samples: int = Field(gt=0)
    repetitions: Optional[int] = Field(default=None, gt=0, le=500)
    seeds: Optional[list[int]] = Field(default=None, min_length=1)

    # Seed or repetitions
    @model_validator(mode="after")
    def seed_rep_check(self) -> Self:
        if self.seeds is not None:
            seen = []
            for seed in self.seeds:
                if seed in seen:
                    raise ValueError(
                        f"Each provided seed for a dataset must be unique. Found duplicate seeds: {seed}."
                    )
                seen.append(seed)
            # set repetitions
            if self.repetitions is None:
                self.repetitions = len(self.seeds)
            else:
                if len(self.seeds) != self.repetitions:
                    raise ValueError(
                        f"The number of seeds provided must be equal to the number of repetitions. Provided {len(self.seeds)} for {self.repetitions} repetitions."
                    )
        elif self.repetitions is not None:
            self.seeds = random.sample(range(0, 10001), self.repetitions)
        else:
            self.repetitions = 1
            self.seeds = [random.randint(0, 10000)]
        return self

    @model_validator(mode="after")
    def feature_uniqueness_check(self) -> Self:
        names = []
        for f in self.features:
            if f.name in names:
                raise ValueError(
                    f"Feature names must be unique for each feature. Found duplicate feature {f.name}."
                )
            names.append(f.name)
        return self

    @model_validator(mode="after")
    def drift_max_range_check(self) -> Self:
        # CD
        if len(self.class_func.drift_defs) > 0:
            drift = self.class_func.drift_defs[
                -1
            ]  # drifts are strictly rising and not overlaping, it's enough to check the last one
            end = drift.center + math.ceil(drift.window / 2)  # conservative, round up
            if end >= self.samples:
                raise ValueError(
                    f"Each drift must finish before the last sample. Concept drift centered at {drift.center}, ending at {end}, exceeds the number of samples {self.samples}"
                )

        # DD
        for f in self.features:
            if len(f.data_dist.drift_defs) > 0:
                drift = f.data_dist.drift_defs[-1]
                end = drift.center + math.ceil(
                    drift.window / 2
                )  # conservative, round up
                if end >= self.samples:
                    raise ValueError(
                        f"Each drift must finish before the last sample. Data drift for feature {f.name}, centered at {drift.center}, ending at {end}, exceeds the number of samples {self.samples}"
                    )

        return self
