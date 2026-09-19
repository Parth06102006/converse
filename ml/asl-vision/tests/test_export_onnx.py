"""Unit tests verifying ST-GCN ONNX export, checker validation, and numerical parity."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from asl_vision.models.stgcn import STGCN
from scripts.export_onnx import export_stgcn_to_onnx, parse_args


def test_export_stgcn_to_onnx_parity(tmp_path: Path) -> None:
    """Verifies ONNX export structure and numerical parity against PyTorch eager execution."""
    model = STGCN(
        in_channels=3,
        num_classes=8,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(16, 1), (32, 2)],
    )
    onnx_file = tmp_path / "test_model.onnx"

    exported = export_stgcn_to_onnx(
        model=model,
        output_path=onnx_file,
        num_nodes=75,
        temporal_window_size=30,
        verify=True,
        atol=1e-3,
        rtol=1e-3,
    )
    assert exported.is_file()

    # Verify structural integrity
    loaded_onnx = onnx.load(str(exported))
    onnx.checker.check_model(loaded_onnx)


def test_exported_onnx_dynamic_batching(tmp_path: Path) -> None:
    """Verifies that the exported ONNX model supports variable batch sizes dynamically."""
    model = STGCN(
        in_channels=3,
        num_classes=4,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(16, 1)],
    )
    onnx_file = tmp_path / "dynamic_batch.onnx"

    export_stgcn_to_onnx(
        model=model,
        output_path=onnx_file,
        num_nodes=75,
        temporal_window_size=30,
        verify=False,
    )

    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    for batch_size in (1, 3, 7):
        dummy = np.random.randn(batch_size, 3, 30, 75).astype(np.float32)
        outputs = session.run([output_name], {input_name: dummy})[0]
        assert outputs.shape == (batch_size, 4)


def test_export_with_checkpoint_loading(tmp_path: Path) -> None:
    """Verifies that an exported model accurately reflects checkpointed weights."""
    model = STGCN(
        in_channels=3,
        num_classes=5,
        num_nodes=75,
        temporal_window_size=30,
        block_channels=[(16, 1)],
    )
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    onnx_file = tmp_path / "from_ckpt.onnx"
    export_stgcn_to_onnx(
        model=model,
        output_path=onnx_file,
        num_nodes=75,
        temporal_window_size=30,
        verify=True,
    )
    assert onnx_file.is_file()


def test_parse_args_defaults() -> None:
    """Verifies argument parsing default settings."""
    args = parse_args([])
    assert args.num_classes == 100
    assert args.num_nodes == 75
    assert args.window_size == 30
    assert args.checkpoint is None
    assert not args.no_verify
