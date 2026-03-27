"""Helper for downloading pretrained U²-Net models and exporting ONNX artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

HUB_REPO = "xuebinqin/U-2-Net"
AVAILABLE_VARIANTS = {"u2net", "u2netp"}


def _normalize_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    if "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        return {key.replace("module.", ""): value for key, value in state.items()}
    raise RuntimeError("Checkpoint format not recognized")


def build_model(
    variant: str,
    checkpoint: Optional[Path] = None,
    pretrained: bool = True,
    force_reload: bool = False,
    repo_path: Optional[Path] = None,
) -> torch.nn.Module:
    if variant not in AVAILABLE_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Available options: {sorted(AVAILABLE_VARIANTS)}")
    logger.info("Loading %s (pretrained=%s, force_reload=%s)", variant, pretrained, force_reload)
    repo_or_dir = str(repo_path) if repo_path is not None else HUB_REPO
    model = torch.hub.load(
        repo_or_dir,
        variant,
        pretrained=pretrained,
        force_reload=force_reload,
        verbose=False,
        trust_repo=True,
        source="local" if repo_path is not None else "github",
    )

    if checkpoint is not None:
        logger.info("Loading checkpoint from %s", checkpoint)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(_normalize_state_dict(state_dict))

    model.eval()
    return model


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_size: tuple[int, int] = (320, 320),
    opset: int = 18,
) -> None:
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["mask"],
        dynamo=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained U²-Net variants and export them as ONNX artifacts"
    )
    parser.add_argument(
        "--variant",
        choices=sorted(AVAILABLE_VARIANTS),
        default="u2net",
        help="Model variant to download/export (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("u2net.onnx"),
        help="Output path for the exported ONNX model",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[320, 320],
        metavar=("HEIGHT", "WIDTH"),
        help="Height and width for the dummy input tensor",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to target",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional local checkpoint to load after downloading the pretrained variant",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        help="Local U²-Net repository to load instead of downloading from torch.hub",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force torch.hub to re-download the requested variant",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Skip downloading pretrained weights (requires --checkpoint)",
    )
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.pretrained and args.checkpoint is None:
        parser = argparse.ArgumentParser()
        raise SystemExit("--no-pretrained requires --checkpoint to be provided")

    model = build_model(
        variant=args.variant,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        force_reload=args.force_reload,
        repo_path=args.repo,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.output, tuple(args.size), args.opset)
    logger.info("Exported U²-Net ONNX model to %s", args.output)


if __name__ == "__main__":
    main()
