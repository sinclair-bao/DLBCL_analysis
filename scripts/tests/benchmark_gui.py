"""Read-only GUI timing probe for a patient; no outputs are written."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from time import sleep

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "gui", ROOT / "longitudinal", ROOT / "common", ROOT):
    sys.path.insert(0, str(folder))

from main_window import MainWindow  # noqa: E402
from session import load_session  # noqa: E402


def main() -> None:
    app = QApplication([])
    start = perf_counter()
    win = MainWindow(ROOT.parent / "data" / "interim", ROOT.parent / "data" / "processed")
    print(f"construct: {perf_counter() - start:.3f}s", flush=True)
    patient_id = next((arg for arg in sys.argv[1:] if not arg.startswith("--")),
                      win.catalog.patient_ids()[0])
    if "--gpu" in sys.argv:
        from render_backend import gpu_available
        print(f"GPU available: {gpu_available()}", flush=True)
        win.ortho.combo_render.setCurrentIndex(win.ortho.combo_render.findData("gpu"))
    start = perf_counter()
    win._on_patient(patient_id)
    print(f"open patient {patient_id}: {perf_counter() - start:.3f}s", flush=True)
    start = perf_counter()
    while win._prefetch_worker is not None or win._prefetch_pending:
        app.processEvents()
        sleep(0.01)
    app.processEvents()
    print(f"background MIPs ready: {perf_counter() - start:.3f}s", flush=True)
    start = perf_counter()
    win._refresh_evolution()
    print(f"refresh MIPs: {perf_counter() - start:.3f}s", flush=True)
    feature_csv = win.catalog.processed_root / patient_id / "longitudinal_features.csv"
    start = perf_counter()
    win._feature_csv_current(feature_csv, patient_id,
                             load_session(win.catalog.processed_root, patient_id))
    print(f"check historical features: {perf_counter() - start:.3f}s", flush=True)
    dates = win.catalog.get_patient(patient_id).study_dates()
    session = load_session(win.catalog.processed_root, patient_id)
    for date in dates[:3]:
        start = perf_counter()
        vol = win._load_vol(patient_id, date, session.baseline)
        print(f"cached volume {date}: {perf_counter() - start:.3f}s", flush=True)
        if vol is not None:
            start = perf_counter()
            win.ortho.set_volumes(vol)
            print(f"bind/render {date}: {perf_counter() - start:.3f}s", flush=True)
        start = perf_counter()
        win._on_study(patient_id, date)
        print(f"switch to {date}: {perf_counter() - start:.3f}s", flush=True)
    for date in dates[:2]:
        start = perf_counter()
        win._on_study(patient_id, date)
        print(f"revisit {date}: {perf_counter() - start:.3f}s", flush=True)
    if win.ortho.volumes() is not None:
        start = perf_counter()
        for step in range(15):
            win.ortho.slider_k.setValue((win.ortho.slider_k.value() + 1) %
                                        (win.ortho.slider_k.maximum() + 1))
        print(f"15 axial slices: {perf_counter() - start:.3f}s", flush=True)
    win.close()
    app.quit()


if __name__ == "__main__":
    main()
