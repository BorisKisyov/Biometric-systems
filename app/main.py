from __future__ import annotations

import json
import re

import gradio as gr

from .fingerprint_service import (
    FingerprintError,
    current_model_status,
    get_service,
    warm_up_models,
)
from .settings import ACCURATE_DEFAULT_THRESHOLD, FAST_DEFAULT_THRESHOLD, PORT, PRECISION


MODE_CONFIG = {
    "Fast": {
        "threshold": FAST_DEFAULT_THRESHOLD,
        "try_rotations": False,
        "max_precision": min(PRECISION, 20),
        "help": (
            "Fast mode checks the uploaded orientation only. It is the best choice "
            "for quick testing and repeated comparisons."
        ),
    },
    "Accurate": {
        "threshold": ACCURATE_DEFAULT_THRESHOLD,
        "try_rotations": True,
        "max_precision": PRECISION,
        "help": (
            "Accurate mode also tries 90-degree rotations. It is slower, but more "
            "robust when the fingerprint orientation is inconsistent."
        ),
    },
}


def _normalize_mode_name(mode_name) -> str:
    if isinstance(mode_name, str) and mode_name in MODE_CONFIG:
        return mode_name
    if isinstance(mode_name, dict):
        for key in ("value", "label", "selected"):
            value = mode_name.get(key)
            if isinstance(value, str) and value in MODE_CONFIG:
                return value
        for value in mode_name.values():
            if isinstance(value, str) and value in MODE_CONFIG:
                return value
    return "Fast"


def _status_markdown() -> str:
    status = current_model_status()
    if status["ready"]:
        return (
            "Model status: ready. Models are preloaded at startup. "
            "Use Fast mode for quick testing and Accurate mode for harder pairs."
        )
    missing = ", ".join(status["missing"])
    return (
        "Model status: missing weights. Missing files: "
        f"`{missing}`. Start with Docker or run `python -m app.download_models`."
    )


def _mode_help(mode_name: str) -> str:
    selected_mode = _normalize_mode_name(mode_name)
    return MODE_CONFIG[selected_mode]["help"]


def _mode_changed(mode_name: str):
    selected_mode = _normalize_mode_name(mode_name)
    return (
        gr.update(value=MODE_CONFIG[selected_mode]["threshold"]),
        _mode_help(selected_mode),
    )


def _comparison_failure(exc: FingerprintError, mode_name: str, threshold: float):
    reason = str(exc)
    details = {
        "error": reason,
        "mode": mode_name,
        "threshold": float(threshold),
    }
    lines = [
        "## Comparison failed",
        f"- Mode: `{mode_name}`",
        f"- Decision threshold: `{float(threshold):.2f}`",
    ]

    if "minutiae" in reason and "at least" in reason:
        counts = [int(value) for value in re.findall(r"Only (\\d+) minutiae", reason)]
        minimum_match = re.search(r"at least (\\d+) are needed", reason)
        best_count = max(counts) if counts else None
        required = int(minimum_match.group(1)) if minimum_match else None
        if best_count is not None:
            details["best_detected_minutiae"] = best_count
        if required is not None:
            details["required_minutiae"] = required

        lines.extend(
            [
                "- Cause: one of the uploaded fingerprint images does not have enough usable ridge detail for the model.",
                (
                    f"- Best extraction attempt found `{best_count}` minutiae, but the model needs at least `{required}`."
                    if best_count is not None and required is not None
                    else f"- Reason: {reason}"
                ),
                "- Accurate mode already tried 0, 90, 180, and 270 degree rotations.",
                "- The threshold is not the cause here; matching never started.",
                "- Tip: use a clearer, less cropped image with stronger ridge contrast.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Reason: {reason}",
                "- Tip: try a cleaner fingerprint image or switch to Fast mode for a quick sanity check.",
            ]
        )

    summary = "\n".join(lines)
    details_text = json.dumps(details, indent=2, ensure_ascii=False)
    return summary, details_text, None, None


def compare_fingerprints(image_a, image_b, threshold: float, mode_name: str):
    if image_a is None or image_b is None:
        raise gr.Error("Please upload both fingerprint images.")

    selected_mode = _normalize_mode_name(mode_name)
    mode = MODE_CONFIG[selected_mode]

    try:
        service = get_service()
        result = service.compare(
            image_a=image_a,
            image_b=image_b,
            threshold=float(threshold),
            try_rotations=bool(mode["try_rotations"]),
            max_precision=int(mode["max_precision"]),
            mode_name=selected_mode,
        )
    except FingerprintError as exc:
        return _comparison_failure(exc, selected_mode, float(threshold))

    score = result["score"]
    summary = (
        f"## {result['label']}\n"
        f"- Mode: `{selected_mode}`\n"
        f"- Score: `{score:.4f}`\n"
        f"- Decision threshold: `{float(threshold):.2f}`\n"
        f"- Rule: higher score means more likely the same finger"
    )
    details_text = json.dumps(result["details"], indent=2)
    return summary, details_text, result["overlay_a"], result["overlay_b"]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Fingerprint Verification Demo") as demo:
        gr.Markdown(
            """
# Fingerprint Verification Demo

Compare two fingerprint images with a pretrained deep learning pipeline based on
FingerFlow, MinutiaeNet, and VerifyNet.
"""
        )
        gr.Markdown(_status_markdown())

        with gr.Row():
            image_a = gr.Image(label="Fingerprint A", type="numpy")
            image_b = gr.Image(label="Fingerprint B", type="numpy")

        with gr.Row():
            mode = gr.Radio(
                choices=list(MODE_CONFIG.keys()),
                value="Fast",
                label="Comparison mode",
            )
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=FAST_DEFAULT_THRESHOLD,
                step=0.01,
                label="Match threshold",
            )

        mode_help = gr.Markdown(_mode_help("Fast"))

        compare_button = gr.Button("Compare fingerprints", variant="primary")

        summary = gr.Markdown()
        details = gr.Code(label="Technical details", language="json")

        with gr.Row():
            overlay_a = gr.Image(label="Fingerprint A with selected minutiae", type="numpy")
            overlay_b = gr.Image(label="Fingerprint B with selected minutiae", type="numpy")

        mode.change(
            fn=_mode_changed,
            inputs=[mode],
            outputs=[threshold, mode_help],
        )

        compare_button.click(
            fn=compare_fingerprints,
            inputs=[image_a, image_b, threshold, mode],
            outputs=[summary, details, overlay_a, overlay_b],
        )

    return demo


if __name__ == "__main__":
    if current_model_status()["ready"]:
        print("Preloading fingerprint models...")
        warm_up_models()
        print("Fingerprint models are ready.")
    build_app().launch(server_name="0.0.0.0", server_port=PORT)
