# Biometric Systems - Fingerprint Verification Demo

A coursework project for the course `Biometric Systems`, focused on fingerprint
verification with deep learning and a pretrained model pipeline.

This repository contains a Dockerized demo based on `FingerFlow`,
`MinutiaeNet`, and `VerifyNet`. The app compares two fingerprint images and
returns:

- a similarity score
- a same/different decision
- visual overlays with selected minutiae and the detected core

The UI includes two comparison modes:

- `Fast` for quicker testing
- `Accurate` for rotation-aware matching

## Important note about the model files

The large pretrained weights are **not supposed to be committed to GitHub**.
They are downloaded automatically when the app starts in Docker, or manually via
the provided local setup script.

The `models/` folder is intentionally ignored by Git except for
`models/README.md`.

Why:

- GitHub rejects some of the model files because they are too large
- the files are downloaded artifacts, not source code
- keeping them out of the repository makes cloning and pushing much easier

If you really want to version large binary model files, use Git LFS. For this
project that is unnecessary because the downloader already fetches the exact
weights.

## Requirements

The local Python setup uses these package versions:

```text
fingerflow==3.0.1
gradio==4.44.1
gdown==5.2.0
huggingface_hub==0.25.2
pydantic==2.10.6
tensorflow==2.12.0
keras==2.12.0
numpy==1.23.5
```

## Folder layout

- `app/` application code
- `data/fingerprints/` sample fingerprint data
- `models/` downloaded pretrained model weights
- `scripts/` helper PowerShell scripts for local setup

## Included sample dataset

The repository includes a small public BMP fingerprint dataset under:

- `data/fingerprints/fvs/raw`
- `data/fingerprints/fvs/by_finger`

Source:

- curated index: https://github.com/robertvazan/fingerprint-datasets
- original dataset page: https://fvs.sourceforge.net/
- direct archive used: https://fvs.sourceforge.net/fingerprint_bitmaps.zip

This sample set contains `168 BMP` images arranged as `21 fingers x 8 impressions`.

## Fingerprint image format

Recommended:

- `PNG` or `BMP`
- grayscale if possible
- one fingerprint per file
- two or more impressions of the same finger if you want to test genuine pairs

## Run with Docker

1. Install Docker Desktop.
2. From the project root, run:

```powershell
docker compose up --build
```

3. Open:

```text
http://localhost:7860
```

By default, `docker-compose.yml` sets `AUTO_DOWNLOAD_MODELS=1`, so the
container downloads the pretrained weights into `./models` on first start.

## Run locally with PowerShell

From the project root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-local.ps1
```

Then start the app with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-local.ps1
```

What `setup-local.ps1` does:

- creates `.venv` if needed
- installs `requirements.txt`
- downloads the pretrained model files into `models/`

## Notes about the score

The matcher outputs a score in the range `0-1`. Higher means a stronger match.
The default threshold is `0.70`, and you can adjust it in the UI.

Performance tips:

- `Fast` mode is the best default for quick testing
- `Accurate` mode also tests rotated versions of the fingerprints
- repeated comparisons of the same uploaded images are faster because extracted
  candidates are cached in memory

## Push this project to GitHub

Do **not** upload the downloaded model files through the GitHub website.
They are too large and they do not belong in source control.

If your local branch already contains those files in a commit, the cleanest fix
is to rebuild the local `main` branch from `origin/main` while keeping your
current working tree files:

```powershell
git fetch origin
git reset --mixed origin/main
git add .gitignore README.md Dockerfile docker-compose.yml entrypoint.sh requirements.txt app data models\README.md scripts
git commit -m "Add fingerprint verification demo without model weights"
git push origin main
```

Why these commands work:

- `git reset --mixed origin/main` moves your branch pointer back to the current
  remote branch without deleting your local files
- `git add ...` stages the project files again
- the new `.gitignore` prevents `models/` weights from being added
- the new commit no longer contains the oversized binary files

If you prefer, you can first inspect what will be committed with:

```powershell
git status
```

## Sources used

- FingerFlow README: https://github.com/jakubarendac/fingerflow
- FingerFlow package: https://pypi.org/project/fingerflow/
- MinutiaeNet reference mentioned by FingerFlow:
  https://github.com/luannd/MinutiaeNet
