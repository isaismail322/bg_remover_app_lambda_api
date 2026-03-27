import base64
import io
import json
import logging
import os

from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Resampling

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/opt/onnx-models/u2net.onnx")
INPUT_SIZE = (320, 320)

_ort_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    global _ort_session
    if _ort_session is None:
        logger.debug("Loading ONNX model from %s", MODEL_PATH)
        _ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _ort_session


def _parse_body(event: Dict[str, Any]) -> str:
    raw_body = event.get("body")
    if raw_body is None:
        raise ValueError("Request body is required")
    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body).decode("utf-8")
    elif isinstance(raw_body, bytes):
        raw_body = raw_body.decode("utf-8")
    payload = json.loads(raw_body)
    image_b64 = payload.get("image")
    if not image_b64:
        raise ValueError("Payload must include an 'image' field containing a base64 string")
    return image_b64


def _preprocess(image_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int], Image.Image]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    resized = image.resize(INPUT_SIZE, Resampling.LANCZOS)
    array = np.asarray(resized).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    normalized = (array - mean) / std
    tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
    return tensor, original_size, image


def _postprocess(mask_tensor: np.ndarray, original_size: Tuple[int, int]) -> Image.Image:
    mask = mask_tensor[0][0]
    mask_min, mask_max = mask.min(), mask.max()
    mask = (mask - mask_min) / (mask_max - mask_min + 1e-8)
    mask_img = Image.fromarray((mask * 255).clip(0, 255).astype("uint8"))
    mask_img = mask_img.resize(original_size, Resampling.LANCZOS)
    return mask_img


def _to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _error_response(message: str, status_code: int = 400) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        image_b64 = _parse_body(event)
        image_bytes = base64.b64decode(image_b64)
        tensor, original_size, image = _preprocess(image_bytes)
        session = _get_session()
        input_name = session.get_inputs()[0].name
        raw_mask = session.run(None, {input_name: tensor})[0]
        mask_image = _postprocess(raw_mask, original_size)
        result_base64 = _to_base64(image.convert("RGBA"))
        mask_base64 = _to_base64(mask_image)
        response_body = {
            "result": {
                "image": result_base64,
                "mask": mask_base64,
            }
        }
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response_body),
        }
    except ValueError as exc:
        logger.warning("Validation failed: %s", exc)
        return _error_response(str(exc), status_code=400)
    except Exception as exc:  # noqa: BLE004
        logger.exception("Background removal failed")
        return _error_response("Internal server error", status_code=500)
