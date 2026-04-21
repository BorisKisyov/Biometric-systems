from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import gdown

from .settings import MODEL_DIR, MODEL_SPECS, ModelSpec


def ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def missing_models() -> list[ModelSpec]:
    ensure_model_dir()
    return [spec for spec in MODEL_SPECS.values() if not spec.path.exists()]


def download_model(spec: ModelSpec, force: bool = False) -> Path:
    ensure_model_dir()
    destination = spec.path
    if destination.exists() and not force:
        return destination

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    if "dropbox.com" in spec.url:
        urllib.request.urlretrieve(spec.url, tmp_path)
    else:
        gdown.download(spec.url, str(tmp_path), quiet=False, fuzzy=True)

    tmp_path.replace(destination)
    return destination


def download_all(force: bool = False) -> list[Path]:
    downloaded = []
    for spec in MODEL_SPECS.values():
        downloaded.append(download_model(spec, force=force))
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pretrained FingerFlow weights for the demo."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload model files even if they already exist.",
    )
    args = parser.parse_args()

    paths = download_all(force=args.force)
    print("Downloaded or verified model files:")
    for path in paths:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f" - {path.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
