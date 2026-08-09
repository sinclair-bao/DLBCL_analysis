# Data dictionary

## `data/raw/`

| Path | Description |
|------|-------------|
| `DICOM/` | Original DICOM batch |
| `DICOMDIS`–`DICOMDIY` | Additional DICOM batches |
| `DICOMDIR*` | DICOMDIR catalog files |
| `DIR_INFO.TXT` | Directory listing / import notes |

## Conventions

- Subject / patient folders should use stable study IDs.
- Do not modify files under `raw/`; write derivatives to `interim/` or `processed/`.
