import datetime
from enum import Enum


class SecondTimedelta(datetime.timedelta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v) -> "SecondTimedelta":
        if not any(isinstance(v, t) for t in (int, float)):
            raise AssertionError(f"don't know how to convert {v=} into seconds")
        if not v > 0:
            raise AssertionError(f"{v=} but seconds should be larger than 0")
        return cls(seconds=v)


class PhaseDirection(str, Enum):
    i = "i"
    j = "j"
    k = "k"
    i_ = "i-"
    j_ = "j-"
    k_ = "k-"


class Part(str, Enum):
    mag = "mag"
    phase = "phase"
    real = "real"
    imag = "imag"


class Suffix(str, Enum):
    bold = "bold"
    cbv = "cbv"
    dwi = "dwi"
    epi = "epi"
    fieldmap = "fieldmap"
    magnitude = "magnitude"
    magnitude1 = "magnitude1"
    magnitude2 = "magnitude2"
    phase1 = "phase1"
    phase2 = "phase2"
    phasediff = "phasediff"
    physio = "physio"
    sbref = "sbref"
    stim = "stim"
    T1w = "T1w"
    T2w = "T2w"
    T2starw = "T2starw"


SUFFIXES = "|".join(suffix.value for suffix in Suffix)
