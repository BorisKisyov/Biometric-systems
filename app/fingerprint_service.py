from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import signal
from skimage import filters as skimage_filters

from .settings import (
    CANDIDATE_CACHE_SIZE,
    CLASS_NAME_TO_ID,
    MODEL_SPECS,
    PRECISION,
    SUPPORTED_PRECISIONS,
    verify_model_key,
)


class FingerprintError(RuntimeError):
    pass


@dataclass
class ExtractionCandidate:
    angle: int
    image_bgr: np.ndarray
    overlay_rgb: np.ndarray
    feature_table: pd.DataFrame
    minutiae_count: int
    available_count: int
    core_score: float


def _ensure_three_channel_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise FingerprintError("Unsupported image format. Use a PNG or BMP fingerprint image.")


def _rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image.copy()
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def _best_core(core_df: pd.DataFrame) -> pd.Series:
    if core_df is None or core_df.empty:
        raise FingerprintError("No fingerprint core was detected.")
    return core_df.sort_values("score", ascending=False).iloc[0]


def _normalize_class_value(value: Any) -> float:
    if isinstance(value, str):
        return float(CLASS_NAME_TO_ID.get(value.lower(), 0))
    return float(value)


def _prepare_feature_table(minutiae_df: pd.DataFrame, core: pd.Series) -> pd.DataFrame:
    if minutiae_df is None or minutiae_df.empty:
        raise FingerprintError("No minutiae were extracted from the image.")

    required_columns = {"x", "y", "angle", "score"}
    if not required_columns.issubset(minutiae_df.columns):
        raise FingerprintError("Extractor output is missing required minutiae columns.")

    core_x = (float(core["x1"]) + float(core["x2"])) / 2.0
    core_y = (float(core["y1"]) + float(core["y2"])) / 2.0

    prepared = minutiae_df.copy()
    if "class" not in prepared.columns:
        prepared["class"] = 0
    prepared["class"] = prepared["class"].map(_normalize_class_value)
    prepared["core_distance"] = np.linalg.norm(
        prepared[["x", "y"]].to_numpy(dtype=np.float32)
        - np.array([[core_x, core_y]], dtype=np.float32),
        axis=1,
    )
    prepared = prepared.sort_values("core_distance").reset_index(drop=True)
    min_precision = min(SUPPORTED_PRECISIONS)
    if len(prepared) < min_precision:
        raise FingerprintError(
            f"Only {len(prepared)} minutiae were extracted; at least {min_precision} are needed."
        )
    return prepared


def _build_feature_array(feature_table: pd.DataFrame, precision: int) -> np.ndarray:
    selected = feature_table.head(precision).copy()
    columns = ["x", "y", "angle", "score", "class", "core_distance"]
    return selected[columns].to_numpy(dtype=np.float32)


def _annotate_image(
    image_bgr: np.ndarray, feature_table: pd.DataFrame, core: pd.Series
) -> np.ndarray:
    annotated = image_bgr.copy()
    start = (int(core["x1"]), int(core["y1"]))
    end = (int(core["x2"]), int(core["y2"]))
    center = (
        int((float(core["x1"]) + float(core["x2"])) / 2.0),
        int((float(core["y1"]) + float(core["y2"])) / 2.0),
    )

    cv2.rectangle(annotated, start, end, (0, 0, 255), 2)
    cv2.circle(annotated, center, 5, (0, 0, 255), -1)

    palette = {
        0: (255, 255, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 255, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
    }
    overlay_precision = min(len(feature_table), PRECISION)
    for _, point in feature_table.head(overlay_precision).iterrows():
        point_xy = (int(point["x"]), int(point["y"]))
        color = palette.get(int(point["class"]), (200, 200, 200))
        cv2.circle(annotated, point_xy, 8, color, 1)
        cv2.line(annotated, center, point_xy, (0, 0, 255), 1)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


class FingerprintService:
    def __init__(self) -> None:
        missing = [spec.path.name for spec in MODEL_SPECS.values() if not spec.path.exists()]
        if missing:
            raise FingerprintError(
                "Missing pretrained weights: "
                + ", ".join(missing)
                + ". Run `python -m app.download_models` or start with Docker."
            )

        if not hasattr(signal, "gaussian") and hasattr(signal, "windows"):
            signal.gaussian = signal.windows.gaussian
        if "multichannel" not in inspect.signature(skimage_filters.gaussian).parameters:
            original_gaussian = skimage_filters.gaussian

            def gaussian_compat(image, sigma=1, output=None, mode="nearest",
                                cval=0, preserve_range=False, truncate=4.0,
                                multichannel=None, channel_axis=None, **kwargs):
                if multichannel is not None and channel_axis is None:
                    channel_axis = -1 if multichannel else None
                return original_gaussian(
                    image,
                    sigma=sigma,
                    mode=mode,
                    cval=cval,
                    preserve_range=preserve_range,
                    truncate=truncate,
                    channel_axis=channel_axis,
                    **kwargs,
                )

            skimage_filters.gaussian = gaussian_compat

        from fingerflow.extractor import Extractor
        from fingerflow.matcher import Matcher

        self._extractor = Extractor(
            str(MODEL_SPECS["coarse_net"].path),
            str(MODEL_SPECS["fine_net"].path),
            str(MODEL_SPECS["classify_net"].path),
            str(MODEL_SPECS["core_net"].path),
        )
        self._matcher_cls = Matcher
        self._matchers: dict[int, Any] = {}
        self._candidate_cache: OrderedDict[tuple[str, bool], list[ExtractionCandidate]] = (
            OrderedDict()
        )

    def _get_matcher(self, precision: int):
        matcher = self._matchers.get(precision)
        if matcher is None:
            matcher = self._matcher_cls(
                precision,
                str(MODEL_SPECS[verify_model_key(precision)].path),
            )
            self._matchers[precision] = matcher
        return matcher

    def _choose_precision(self, count_a: int, count_b: int, max_precision: int) -> int:
        available = min(count_a, count_b, max_precision)
        for precision in SUPPORTED_PRECISIONS:
            if precision <= available:
                return precision
        raise FingerprintError(
            "Not enough extracted minutiae for any supported matcher precision."
        )

    def warm_up(self) -> None:
        for precision in SUPPORTED_PRECISIONS:
            self._get_matcher(precision)

    def _image_cache_key(self, image_rgb: np.ndarray, try_rotations: bool) -> tuple[str, bool]:
        contiguous = np.ascontiguousarray(image_rgb)
        digest = hashlib.sha1(contiguous.tobytes()).hexdigest()
        shape_key = "x".join(map(str, contiguous.shape))
        return (f"{shape_key}:{digest}", try_rotations)

    def _cached_candidates(
        self, image_rgb: np.ndarray, try_rotations: bool
    ) -> list[ExtractionCandidate] | None:
        key = self._image_cache_key(image_rgb, try_rotations)
        candidates = self._candidate_cache.get(key)
        if candidates is not None:
            self._candidate_cache.move_to_end(key)
        return candidates

    def _store_candidates(
        self,
        image_rgb: np.ndarray,
        try_rotations: bool,
        candidates: list[ExtractionCandidate],
    ) -> None:
        key = self._image_cache_key(image_rgb, try_rotations)
        self._candidate_cache[key] = candidates
        self._candidate_cache.move_to_end(key)
        while len(self._candidate_cache) > CANDIDATE_CACHE_SIZE:
            self._candidate_cache.popitem(last=False)

    def _extract_candidate(self, image_rgb: np.ndarray, angle: int) -> ExtractionCandidate:
        image_bgr = _ensure_three_channel_bgr(image_rgb)
        extracted = self._extractor.extract_minutiae(image_bgr)
        core = _best_core(extracted["core"])
        feature_table = _prepare_feature_table(extracted["minutiae"], core)
        overlay_rgb = _annotate_image(image_bgr, feature_table, core)
        return ExtractionCandidate(
            angle=angle,
            image_bgr=image_bgr,
            overlay_rgb=overlay_rgb,
            feature_table=feature_table,
            minutiae_count=len(extracted["minutiae"]),
            available_count=len(feature_table),
            core_score=float(core["score"]),
        )

    def _candidates(self, image_rgb: np.ndarray, try_rotations: bool) -> list[ExtractionCandidate]:
        cached = self._cached_candidates(image_rgb, try_rotations)
        if cached is not None:
            return cached

        angles = [0, 90, 180, 270] if try_rotations else [0]
        candidates = []
        errors = []
        for angle in angles:
            rotated = _rotate_image(image_rgb, angle)
            try:
                candidates.append(self._extract_candidate(rotated, angle))
            except FingerprintError as exc:
                errors.append(f"{angle} deg: {exc}")

        if not candidates:
            raise FingerprintError(
                "The fingerprint could not be encoded from any tested rotation. "
                + " | ".join(errors)
            )
        self._store_candidates(image_rgb, try_rotations, candidates)
        return candidates

    def compare(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        threshold: float,
        try_rotations: bool,
        max_precision: int | None = None,
        mode_name: str | None = None,
    ) -> dict[str, Any]:
        candidates_a = self._candidates(image_a, try_rotations)
        candidates_b = self._candidates(image_b, try_rotations)
        precision_cap = min(max_precision or PRECISION, PRECISION)

        best = None
        for candidate_a in candidates_a:
            for candidate_b in candidates_b:
                precision = self._choose_precision(
                    candidate_a.available_count,
                    candidate_b.available_count,
                    precision_cap,
                )
                matcher = self._get_matcher(precision)
                score = float(
                    matcher.verify(
                        _build_feature_array(candidate_a.feature_table, precision),
                        _build_feature_array(candidate_b.feature_table, precision),
                    )
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "precision": precision,
                    }

        if best is None:
            raise FingerprintError("No valid fingerprint comparison could be produced.")

        score = best["score"]
        is_match = score >= threshold
        label = "Same finger" if is_match else "Different finger"
        return {
            "score": score,
            "label": label,
            "match": is_match,
            "overlay_a": best["candidate_a"].overlay_rgb,
            "overlay_b": best["candidate_b"].overlay_rgb,
            "details": {
                "mode": mode_name or ("Accurate" if try_rotations else "Fast"),
                "threshold": threshold,
                "rotation_search": try_rotations,
                "selected_rotation_a": best["candidate_a"].angle,
                "selected_rotation_b": best["candidate_b"].angle,
                "detected_minutiae_a": best["candidate_a"].minutiae_count,
                "detected_minutiae_b": best["candidate_b"].minutiae_count,
                "available_minutiae_a": best["candidate_a"].available_count,
                "available_minutiae_b": best["candidate_b"].available_count,
                "used_minutiae_per_fingerprint": best["precision"],
                "core_score_a": round(best["candidate_a"].core_score, 4),
                "core_score_b": round(best["candidate_b"].core_score, 4),
            },
        }


@lru_cache(maxsize=1)
def get_service() -> FingerprintService:
    return FingerprintService()


def warm_up_models() -> None:
    service = get_service()
    service.warm_up()


def current_model_status() -> dict[str, Any]:
    missing = [spec.path.name for spec in MODEL_SPECS.values() if not spec.path.exists()]
    return {
        "ready": not missing,
        "missing": missing,
    }
