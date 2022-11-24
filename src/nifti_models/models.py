import typing

import pydantic

import nibabel as nb


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
    magic: typing.Literal[b"ni1", b"n+1"]

    @classmethod
    def from_nibabel(cls, header: nb.Nifti1Header) -> typing.Self:
        h = header.structarr
        keys: dict[str, typing.Any] = {name: h[name].tolist() for name in h.dtype.names}  # type: ignore
        return cls(**keys)


class Nifti1Meta(pydantic.BaseModel):
    header: Nifti1Header
    sidecar: dict

    @classmethod
    def from_nibabel_sidecar(
        cls, img: nb.Nifti1Image, sidecar: dict[str, typing.Any]
    ) -> typing.Self:
        img_header: nb.Nifti1Header = img.header  # type: ignore

        return cls(header=Nifti1Header.from_nibabel(img_header), sidecar=sidecar)
