"""Read one existing complete study without changing patient data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "gui", ROOT / "longitudinal", ROOT / "common", ROOT):
    sys.path.insert(0, str(folder))

from catalog import DataCatalog  # noqa: E402
from render_backend import compose_plane_cpu  # noqa: E402
from volume_io import load_volume_set  # noqa: E402


def main() -> None:
    catalog = DataCatalog()
    for patient_id in catalog.patient_ids():
        patient = catalog.get_patient(patient_id)
        for study in patient.studies:
            if study.completeness() != "ok":
                continue
            vol = load_volume_set(study)
            assert vol is not None
            assert vol.ct.shape == vol.pet.shape == vol.native.shape
            assert np.isfinite(vol.affine).all()
            center = tuple(s // 2 for s in vol.ct.shape)
            rendered = compose_plane_cpu(vol, "axial", *center)
            assert rendered.ndim == 3 and rendered.shape[-1] == 3
            print("one complete study loaded and rendered successfully")
            return
    raise RuntimeError("no complete study found")


if __name__ == "__main__":
    main()
