from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models")).resolve()
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data")).resolve()
PORT = int(os.getenv("PORT", "7860"))
PRECISION = int(os.getenv("FINGERPRINT_PRECISION", "30"))
DEFAULT_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.70"))
FAST_DEFAULT_THRESHOLD = float(os.getenv("FAST_MATCH_THRESHOLD", str(DEFAULT_THRESHOLD)))
ACCURATE_DEFAULT_THRESHOLD = float(
    os.getenv("ACCURATE_MATCH_THRESHOLD", str(max(DEFAULT_THRESHOLD, 0.72)))
)
AUTO_DOWNLOAD_MODELS = os.getenv("AUTO_DOWNLOAD_MODELS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SUPPORTED_PRECISIONS = (30, 24, 20, 14, 10)
CANDIDATE_CACHE_SIZE = int(os.getenv("CANDIDATE_CACHE_SIZE", "32"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    source_name: str
    url: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.filename


MODEL_SPECS = {
    "coarse_net": ModelSpec(
        key="coarse_net",
        filename="CoarseNet.h5",
        source_name="Dropbox",
        url="https://www.dropbox.com/s/gppil4wybdjcihy/CoarseNet.h5?dl=1",
    ),
    "fine_net": ModelSpec(
        key="fine_net",
        filename="FineNet.h5",
        source_name="Dropbox",
        url="https://www.dropbox.com/s/k7q2vs9255jf2dh/FineNet.h5?dl=1",
    ),
    "classify_net": ModelSpec(
        key="classify_net",
        filename="ClassifyNet.h5",
        source_name="Google Drive",
        url="https://drive.google.com/file/d/1dfQDW8yxjmFPVu0Ddui2voxdngOrU3rc/view?usp=sharing",
    ),
    "core_net": ModelSpec(
        key="core_net",
        filename="yolo-kernel_best.weights",
        source_name="Google Drive",
        url="https://drive.google.com/file/d/1v091s0eY4_VOLU9BqDXVSaZcFnA9qJPl/view?usp=sharing",
    ),
    "verify_net_10": ModelSpec(
        key="verify_net_10",
        filename="VerifyNet10.h5",
        source_name="Google Drive",
        url="https://drive.google.com/file/d/1cEz3oCYS4JCUiZxpU5o8lYesMOVgR0rt/view?usp=sharing",
    ),
    "verify_net_14": ModelSpec(
        key="verify_net_14",
        filename="VerifyNet14.h5",
        source_name="Google Drive",
        url="https://drive.google.com/file/d/1CI7z1r99AEV6Lrm2bQeGEFmVdQ8colUW/view?usp=sharing",
    ),
    "verify_net_20": ModelSpec(
        key="verify_net_20",
        filename="VerifyNet20.h5",
        source_name="Google Drive",
        url="https://drive.google.com/file/d/1lP1zDHTa7TemWPluv89ueFWCa95RnLF-/view?usp=sharing",
    ),
    "verify_net_24": ModelSpec(
        key="verify_net_24",
        filename="VerifyNet24.h5",
        source_name="Google Drive",
        url="https://drive.google.com/file/d/1h2RwuM1-mgiF4dfwslbgiI7-K8F4aw2A/view?usp=sharing",
    ),
    "verify_net_30": ModelSpec(
        key="verify_net_30",
        filename="VerifyNet30.h5",
        source_name="Google Drive",
        url="https://drive.google.com/file/d/1gQEzJKlCmUqe7Sx-W-6H1w1NGY8M98bX/view?usp=sharing",
    ),
}


CLASS_NAME_TO_ID = {
    "ending": 0,
    "bifurcation": 1,
    "fragment": 2,
    "enclosure": 3,
    "crossbar": 4,
    "other": 5,
}


def verify_model_key(precision: int) -> str:
    return f"verify_net_{precision}"
