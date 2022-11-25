import pathlib
from nifti_models import models

import nibabel as nb
from nibabel.testing import data_path

example_filename = pathlib.Path(data_path) / "example4d.nii.gz"
img = nb.load(example_filename)

header: nb.Nifti1Header = img.header
sidecar = {"TaskName": "rest"}


def test_header():
    assert models.Nifti1Header.from_nibabel(header)


def test_meta():
    assert models.MRIMeta.from_header_sidecar(img.header, sidecar=sidecar)
