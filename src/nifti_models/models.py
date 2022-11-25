import copy
import datetime
import json
import logging
import pathlib
import typing

import pydantic

import numpy as np
import nibabel as nb

from deepdiff import DeepDiff

from . import fields
from . import bids


def _get_bval_bvec(path: pathlib.Path) -> dict[str, list[typing.Any]]:
    bval: list[float] = np.genfromtxt(path.with_suffix(".bval")).tolist()
    bvec: list[list[float]] = np.genfromtxt(path.with_suffix(".bvec")).tolist()
    return {"bval": bval, "bvec": bvec}


class Nifti1Header(pydantic.BaseModel):
    sizeof_hdr: typing.Literal[348]
    data_type: bytes
    db_name: bytes
    extents: int
    session_error: float
    regular: str = pydantic.Field(min_length=1, max_length=1)
    dim_info: int
    dim: pydantic.conlist(item_type=float, min_items=8, max_items=8)  # type: ignore
    intent_p1: float
    intent_p2: float
    intent_p3: float
    intent_code: float
    datatype: float
    bitpix: float
    slice_start: float
    pixdim: pydantic.conlist(item_type=float, min_items=8, max_items=8)  # type: ignore
    vox_offset: float
    scl_slope: float
    scl_inter: float
    slice_end: float
    slice_code: int
    xyzt_units: int
    cal_max: float
    cal_min: float
    slice_duration: float
    toffset: float
    glmax: int
    glmin: int
    descrip: str = pydantic.Field(max_length=80)
    aux_file: str = pydantic.Field(max_length=24)
    qform_code: float
    sform_code: float
    quatern_b: float
    quatern_c: float
    quatern_d: float
    qoffset_x: float
    qoffset_y: float
    qoffset_z: float
    srow_x: pydantic.conlist(item_type=float, min_items=4, max_items=4)  # type: ignore
    srow_y: pydantic.conlist(item_type=float, min_items=4, max_items=4)  # type: ignore
    srow_z: pydantic.conlist(item_type=float, min_items=4, max_items=4)  # type: ignore
    intent_name: str = pydantic.Field(max_length=16)
    magic: bytes

    class Config:
        frozen = True

    @classmethod
    def from_nibabel(cls, header: nb.Nifti1Header) -> "Nifti1Header":
        h = header.structarr
        keys: dict[str, typing.Any] = {name: h[name].tolist() for name in h.dtype.names}  # type: ignore
        return cls(**keys)

    def get_repetition_time_seconds(self) -> float:
        if not self.xyzt_units == 10:
            raise AssertionError(
                f"found {self.xyzt_units=}, but only know how to handle 10"
            )
        if self.pixdim[4] > 50:
            logging.warning(
                f"found repetition time {self.pixdim[4]} seconds, which seems long"
            )

        return self.pixdim[4]


class MRIMeta(pydantic.BaseModel):
    header: Nifti1Header
    Manufacturer: str | None = None
    ManufacturersModelName: str | None = None
    DeviceSerialNumber: str | None = None
    StationName: str | None = None
    SoftwareVersions: str | None = None
    MagneticFieldStrength: str | None = None
    ReceiveCoilName: str | None = None
    ReceiveCoilActiveElements: str | None = None
    GradientSetType: str | None = None
    MRTransmitCoilSequence: str | None = None
    MatrixCoilMode: str | None = None
    CoilCombinationMethod: str | None = None
    PulseSequenceType: str | None = None
    ScanningSequence: str | list[str] | None = None
    SequenceVariant: str | list[str] | None = None
    ScanOptions: str | list[str] | None = None
    SequenceName: str | None = None
    PulseSequenceDetails: str | None = None
    NonlinearGradientCorrection: bool | None = None
    MRAcquisitionType: str | None = None
    MTState: bool | None = None
    MTOffsetFrequency: float | None = None
    MTPulseBandwidth: float | None = None
    MTNumberOfPulses: float | None = None
    MTPulseShape: str | None = None
    MTPulseDuration: fields.SecondTimedelta | None = None
    SpoilingState: bool | None = None
    SpoilingType: str | None = None
    SpoilingRFPhaseIncrement: float | None = None
    SpoilingGradientMoment: float | None = None
    SpoilingGradientDuration: fields.SecondTimedelta | None = None
    NumberShots: float | list[float] | None = None
    ParallelReductionFactorInPlane: int | None = None
    ParallelAcquisitionTechnique: str | None = None
    PartialFourier: float | None = None
    PartialFourierDirection: str | None = None
    EffectiveEchoSpacing: fields.SecondTimedelta | None = None
    MixingTime: fields.SecondTimedelta | None = None
    PhaseEncodingDirection: fields.PhaseDirection | None = None
    TotalReadoutTime: fields.SecondTimedelta | None = None
    EchoTime: fields.SecondTimedelta | None = None
    InversionTime: fields.SecondTimedelta | None = None
    SliceTiming: list[float] | None = None
    SliceEncodingDirection: fields.PhaseDirection | None = None
    DwellTime: fields.SecondTimedelta | None = None
    FlipAngle: float | list[float] | None = None
    NegativeContrast: bool | None = None
    MultibandAccelerationFactor: int | None = None
    AnatomicalLandmarkCoordinates: list[dict[str, list[int]]] | None = None
    B0FieldIdentifier: str | list[str] | None = None
    B0FieldSource: str | list[str] | None = None
    InstitutionName: str | None = None
    InstitutionAddress: str | None = None
    InstitutionalDepartmentName: str | None = None
    RepetitionTime: fields.SecondTimedelta | None = None

    # fieldmap
    IntendedFor: list[pathlib.Path] | None = None
    EchoTime1: fields.SecondTimedelta | None = None
    EchoTime2: fields.SecondTimedelta | None = None
    Units: typing.Literal["Hz", "rad/s", "T"] | None = None

    # DWI
    bval: list[float] | None = None
    bvec: list[list[float]] | None = None

    # Task
    TaskName: str | None = None
    NumberOfVolumesDiscardedByScanner: pydantic.PositiveInt | None = None
    NumberOfVolumesDiscardedByUser: pydantic.PositiveInt | None = None
    DelayTime: fields.SecondTimedelta | None = None
    AcquisitionDuration: fields.SecondTimedelta | None = None
    DelayAfterTrigger: fields.SecondTimedelta | None = None
    Instructions: str | None = None
    TaskDescription: str | None = None
    CogAtlasID: str | None = None
    CogPOID: str | None = None

    # Anat
    ContrastBolusIngredient: str | None = None
    RepetitionTimeExcitation: fields.SecondTimedelta | None = None
    RepetitionTimePreparation: fields.SecondTimedelta | list[
        fields.SecondTimedelta
    ] | None = None

    # extra bids info
    entities: bids.Entities | None = None

    # from dcm2nii https://github.com/rordenlab/dcm2niix/tree/master/BIDS
    ## https://github.com/rordenlab/dcm2niix/tree/master/BIDS#global-constants
    Modality: str | None = None
    ConversionSoftware: str | None = None
    ConversionSoftwareVersion: str | None = None

    ## https://github.com/rordenlab/dcm2niix/tree/master/BIDS#global-series-information
    BodyPartExamined: str | None = None
    PatientPosition: str | None = None
    ProcedureStepDescription: str | None = None
    SeriesDescription: str | None = None
    ProtocolName: str | None = None
    ImageType: list[str] | None = None
    AcquisitionTime: datetime.time | None = None
    AcquisitionNumber: int | None = None
    ImageComments: str | None = None

    ## https://github.com/rordenlab/dcm2niix/tree/master/BIDS#modality-magnetic-resonance-imaging
    AcquisitionMatrixPE: int | None = None
    DerivedVendorReportedEchoSpacing: fields.SecondTimedelta | None = None
    EchoNumber: int | None = None
    EchoTrainLength: int | None = None
    EstimatedEffectiveEchoSpacing: fields.SecondTimedelta | None = None
    EstimatedTotalReadoutTime: fields.SecondTimedelta | None = None
    ImageOrientationPatientDICOM: list[float] | None = None
    ImagingFrequency: float | None = None
    InPlanePhaseEncodingDirectionDICOM: typing.Literal["ROW", "COL"] | None = None
    NumberOfAverages: int | None = None
    ParallelReductionOutOfPlane: int | None = None
    PercentPhaseFOV: int | None = None
    PercentSampling: int | None = None
    PhaseEncodingSteps: int | None = None
    PixelBandwidth: int | None = None
    RepetitionTimeInversion: fields.SecondTimedelta | None = None
    SAR: float | None = None
    SliceThickness: float | None = None
    SpacingBetweenSlices: float | None = None

    ## https://github.com/rordenlab/dcm2niix/tree/master/BIDS#manufacturer-general-electric
    PulseSequenceName: str | None = None
    InternalPulseSequenceName: str | None = None
    PhaseEncodingPolarityGE: typing.Literal["Flipped", "Unflipped"] | None = None
    NumberOfPointsPerArm: int | None = None
    NumberOfArms: int | None = None
    NumberOfExcitations: int | None = None
    PrescanReuseString: str | None = None

    ## https://github.com/rordenlab/dcm2niix/tree/master/BIDS#manufacturer-philips
    TriggerDelayTime: fields.SecondTimedelta | None = None
    PhilipsRWVSlope: float | None = None
    PhilipsRWVIntercept: float | None = None
    PhilipsRescaleSlope: float | None = None
    PhilipsRescaleIntercept: float | None = None
    PhilipsScaleSlope: float | None = None
    UsePhilipsFloatNotDisplayScaling: bool | None
    PartialFourierEnabled: typing.Literal["YES", "NO"] | None = None
    PhaseEncodingStepsNoPartialFourier: int | None = None
    WaterFatShift: float | None = None

    ## https://github.com/rordenlab/dcm2niix/tree/master/BIDS#manufacturer-siemens-magnetic-resonance-imaging-v
    ## +
    ## https://github.com/rordenlab/dcm2niix/tree/master/BIDS#manufacturer-siemens-magnetic-resonance-imaging-xa
    Interpolation2D: str | None = None
    Interpolation3D: str | None = None
    BaseResolution: int | None = None
    ShimSetting: list[float] | None = None
    DiffusionScheme: typing.Literal["Monopolor", "Bipolar"] | None = None
    DelayTime: fields.SecondTimedelta | None = None
    TxRefAmp: float | None = None
    PhaseResolution: float | None = None
    PhaseOversampling: float | None = None
    CoilString: str | None = None
    FmriExternalInfo: str | None = None
    WipMemBlock: str | None = None
    AveragesDouble: float | None = None
    AccelFact3D: float | None = None
    RefLinesPE: int | None = None
    ConsistencyInfo: str | None = None
    BandwidthPerPixelPhaseEncode: float | None = None
    ImageOrientationText: str | None = None

    class Config:
        frozen = True

    @classmethod
    def from_header_sidecar(
        cls, img_header: nb.Nifti1Header, sidecar: dict[str, typing.Any]
    ):
        j = copy.deepcopy(sidecar)
        header = Nifti1Header.from_nibabel(img_header)
        j.update({"RepetitionTime": header.get_repetition_time_seconds()})
        return cls(header=header, **j)

    @classmethod
    def from_sidecarpath(
        cls, sidecar: pathlib.Path, get_data: bool = False
    ) -> "MRIMeta":
        js: dict[str, typing.Any] = json.loads(sidecar.read_text())
        js.update({"entities": bids.Entities.from_path(sidecar)})
        if js.get("entities").suffix == "dwi":  # type: ignore
            js.update(_get_bval_bvec(sidecar))

        img = nb.load(sidecar.with_suffix(".nii.gz"))
        header: nb.Nifti1Header = img.header
        if get_data:
            js.update({"_data": img.get_fdata()})
        return cls.from_header_sidecar(img_header=header, sidecar=js)

    @property
    def affine(self) -> np.ndarray:
        return np.stack([self.header.srow_x, self.header.srow_y, self.header.srow_z])

    def deepdiff(
        self,
        other: "MRIMeta",
        exclude: dict = {
            "header": {
                "quatern_b",
                "quatern_c",
                "quatern_d",
                "qoffset_x",
                "qoffset_y",
                "qoffset_z",
                "srow_x",
                "srow_y",
                "srow_z",
            },
            "AcquisitionTime": True,
            "ConversionSoftware": True,
            "ConversionSoftwareVersion": True,
            "ReceiveCoilActiveElements": True,
            "AcquisitionNumber": True,
            "ImageComments": True,
            "PhilipsRWVSlope": True,
            "PhilipsRWVIntercept": True,
            "PhilipsRescaleSlope": True,
            "PhilipsRescaleIntercept": True,
            "PhilipsScaleSlope": True,
            "WaterFatShift": True,
            "TxRefAmp": True,
            "BodyPartExamined": True,
        },
        **kwargs,
    ) -> DeepDiff:
        return DeepDiff(
            self.dict(
                exclude=exclude,
                exclude_unset=True,
                exclude_defaults=True,
                exclude_none=True,
            ),
            other.dict(
                exclude=exclude,
                exclude_unset=True,
                exclude_defaults=True,
                exclude_none=True,
            ),
            **kwargs,
        )
