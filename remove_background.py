"""Local utility that runs the ONNX U²-Net model over the test images."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Resampling

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_INPUT_SIZE = (320, 320)


def _load_model(model_path: Path) -> ort.InferenceSession:
    logger.info("Loading ONNX model from %s", model_path)
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _list_inputs(directory: Path) -> Iterable[Path]:
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            yield entry


def _preprocess(image: Image.Image, size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    original_size = image.size
    resized = image.resize(size, Resampling.LANCZOS)
    array = np.asarray(resized).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    normalized = (array - mean) / std
    tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
    return tensor, original_size


def _postprocess(mask_tensor: np.ndarray, original_size: Tuple[int, int]) -> Image.Image:
    mask = mask_tensor[0][0]
    mask_min, mask_max = mask.min(), mask.max()
    mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)
    mask_img = Image.fromarray((mask * 255).clip(0, 255).astype("uint8"))
    return mask_img.resize(original_size, Resampling.LANCZOS)


def _run_inference(session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: tensor})[0]


def _composite(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = mask.convert("L")
    rgba.putalpha(alpha)
    return rgba


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the U²-Net ONNX model on a folder of images")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("u2net.onnx"),
        help="Path to the ONNX U²-Net model",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test_images"),
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write cutouts and masks",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=list(DEFAULT_INPUT_SIZE),
        metavar=("HEIGHT", "WIDTH"),
        help="Height and width for the dummy input tensor",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = _load_model(args.model)

    for image_path in _list_inputs(args.input_dir):
        logger.info("Processing %s", image_path.name)
        image = Image.open(image_path).convert("RGB")
        tensor, original_size = _preprocess(image, tuple(args.size))
        mask_tensor = _run_inference(session, tensor)
        mask_image = _postprocess(mask_tensor, original_size)
        cutout = _composite(image, mask_image)

        cutout_path = args.output_dir / f"{image_path.stem}_cutout.png"
        mask_path = args.output_dir / f"{image_path.stem}_mask.png"
        cutout.save(cutout_path)
        mask_image.save(mask_path)
        logger.info("Saved cutout to %s and mask to %s", cutout_path, mask_path)


if __name__ == "__main__":
    main()
