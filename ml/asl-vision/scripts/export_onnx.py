"""ONNX export harness and validation for ST-GCN sign recognition model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from asl_vision.models.stgcn import STGCN


def export_stgcn_to_onnx(
    model: STGCN | torch.nn.Module,
    output_path: str | Path,
    num_nodes: int = 75,
    temporal_window_size: int = 30,
    opset_version: int = 17,
    verify: bool = True,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Path:
    """Export an ST-GCN model instance to ONNX format with dynamic batch axis and verify numerical parity.

    Args:
        model: Instantiated ST-GCN PyTorch model.
        output_path: Target filesystem path for the .onnx artifact.
        num_nodes: Graph node count V.
        temporal_window_size: Sequence frame length T.
        opset_version: ONNX opset version.
        verify: Whether to execute numerical parity verification via onnxruntime.
        atol: Absolute tolerance for parity verification.
        rtol: Relative tolerance for parity verification.

    Returns:
        Path to the saved and verified ONNX artifact.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(
        1,
        model.in_channels if hasattr(model, "in_channels") else 3,
        temporal_window_size,
        num_nodes,
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    # 1. Structural schema validation
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    # 2. Numerical parity verification via ONNX Runtime
    if verify:
        session = ort.InferenceSession(
            str(out_path),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Test with mini-batch of size 2 to verify dynamic batch execution
        test_input = torch.randn(
            2,
            model.in_channels if hasattr(model, "in_channels") else 3,
            temporal_window_size,
            num_nodes,
            dtype=torch.float32,
        )

        with torch.no_grad():
            torch_logits = model(test_input).cpu().numpy()

        ort_logits = session.run(
            [output_name],
            {input_name: test_input.numpy()},
        )[0]

        np.testing.assert_allclose(
            torch_logits,
            ort_logits,
            rtol=rtol,
            atol=atol,
            err_msg="PyTorch eager logits and ONNX Runtime logits diverged beyond tolerance.",
        )

    return out_path


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export PyTorch ST-GCN model to ONNX runtime format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch .pt model checkpoint (random weights if omitted).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/stgcn_wlasl100.onnx",
        help="Destination path for .onnx output file.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="Number of sign gloss classes.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=75,
        help="Number of skeletal graph nodes.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Temporal window frame length.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Target ONNX opset version.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip onnxruntime numerical parity verification.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """CLI entrypoint for ST-GCN ONNX export."""
    parsed = parse_args(args)

    model = STGCN(
        in_channels=3,
        num_classes=parsed.num_classes,
        num_nodes=parsed.num_nodes,
        temporal_window_size=parsed.window_size,
    )

    if parsed.checkpoint is not None:
        ckpt_path = Path(parsed.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        state_dict: dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    exported_path = export_stgcn_to_onnx(
        model=model,
        output_path=parsed.output,
        num_nodes=parsed.num_nodes,
        temporal_window_size=parsed.window_size,
        opset_version=parsed.opset,
        verify=not parsed.no_verify,
    )

    print(f"Successfully exported ST-GCN model to: {exported_path.resolve()}")


if __name__ == "__main__":
    main()
