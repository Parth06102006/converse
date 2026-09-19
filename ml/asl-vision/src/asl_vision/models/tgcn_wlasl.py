"""Temporal Graph Convolutional Network (TGCN) model for WLASL-100 sign recognition.

Implements the multi-head spatial graph convolution with temporal modeling
matching the pretrained WLASL-100 architecture (dxli94/WLASL, sharonn18/tgcn-wlasl).
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter

# 100 canonical ASL glosses in exact WLASL-100 dataset order
WLASL_100_GLOSSES: tuple[str, ...] = (
    "BOOK", "DRINK", "COMPUTER", "BEFORE", "CHAIR", "GO", "CLOTHES", "WHO", "CANDY", "COUSIN",
    "DEAF", "FINE", "HELP", "NO", "THIN", "WALK", "YEAR", "YES", "ALL", "BLACK",
    "COOL", "FINISH", "HOT", "LIKE", "MANY", "MOTHER", "NOW", "ORANGE", "TABLE", "THANKSGIVING",
    "WHAT", "WOMAN", "BED", "BLUE", "BOWLING", "CAN", "DOG", "FAMILY", "FISH", "GRADUATE",
    "HAT", "HEARING", "KISS", "LANGUAGE", "LATER", "MAN", "SHIRT", "STUDY", "TALL", "WHITE",
    "WRONG", "ACCIDENT", "APPLE", "BIRD", "CHANGE", "COLOR", "CORN", "COW", "DANCE", "DARK",
    "DOCTOR", "EAT", "ENJOY", "FORGET", "GIVE", "LAST", "MEET", "PINK", "PIZZA", "PLAY",
    "SCHOOL", "SECRETARY", "SHORT", "TIME", "WANT", "WORK", "AFRICA", "BASKETBALL", "BIRTHDAY", "BROWN",
    "BUT", "CHEAT", "CITY", "COOK", "DECIDE", "FULL", "HOW", "JACKET", "LETTER", "MEDICINE",
    "NEED", "PAINT", "PAPER", "PULL", "PURPLE", "RIGHT", "SAME", "SON", "TELL", "THURSDAY",
)

# 13 MediaPipe upper-body pose joints used in 55-keypoint skeletal layout
UPPER_BODY_POSE_INDICES: tuple[int, ...] = (
    0,   # Nose
    11,  # Left shoulder
    12,  # Right shoulder
    13,  # Left elbow
    14,  # Right elbow
    15,  # Left wrist
    16,  # Right wrist
    23,  # Left hip
    24,  # Right hip
    1,   # Left eye inner
    4,   # Right eye inner
    7,   # Left ear
    8,   # Right ear
)


class GraphConvolution_att(nn.Module):
    """Spatial graph convolution layer with learnable attention matrix."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(55, 55))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GC_Block(nn.Module):
    """Residual graph convolutional block with batch normalization, tanh activation, and dropout."""

    def __init__(self, in_features: int, p_dropout: float = 0.3, is_resi: bool = True) -> None:
        super().__init__()
        self.is_resi = is_resi
        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)
        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, f = x.shape
        y = self.gc1(x)
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        if self.is_resi:
            return y + x
        return y


class GCN_muti_att(nn.Module):
    """Temporal Graph Convolutional Network with multi-head attention for ASL classification."""

    def __init__(
        self,
        input_feature: int = 100,
        hidden_feature: int = 64,
        num_class: int = 100,
        p_dropout: float = 0.3,
        num_stage: int = 20,
        is_resi: bool = True,
    ) -> None:
        super().__init__()
        self.num_stage = num_stage
        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)
        self.gcbs = nn.ModuleList(
            [GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi) for _ in range(num_stage)]
        )
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        y = self.gc1(x)
        y = self.bn1(y.view(b, -1)).view(b, n, -1)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = torch.mean(y, dim=1)
        return self.fc_out(out)


class TGCNWLASLClassifier:
    """Pretrained WLASL-100 Temporal Graph Convolutional Network inference engine."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        num_samples: int = 50,
        min_confidence: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.num_samples = num_samples
        self.min_confidence = min_confidence
        self.device = torch.device(device)
        self.vocab = WLASL_100_GLOSSES

        # Initialize network
        self.model = GCN_muti_att(
            input_feature=num_samples * 2,  # 50 frames * (x, y) = 100
            hidden_feature=64,
            num_class=len(self.vocab),
            p_dropout=0.3,
            num_stage=20,
        )

        # Load weights
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"TGCN checkpoint not found at: {ckpt_p}")

        state_dict = torch.load(str(ckpt_p), map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Rolling temporal buffer maintaining 55 keypoint coordinates
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=num_samples)

    def add_frame(self, extracted: Any) -> None:
        """Extract 55 keypoints (13 upper pose + 21 left hand + 21 right hand) and append to buffer."""
        # Frame shape: (55, 2)
        pts = np.zeros((55, 2), dtype=np.float32)

        # 1. Upper-body pose (13 points)
        if extracted.pose is not None and len(extracted.pose) >= 25:
            for i, idx in enumerate(UPPER_BODY_POSE_INDICES):
                pts[i, 0] = float(extracted.pose[idx, 0])
                pts[i, 1] = float(extracted.pose[idx, 1])

        # 2. Left hand (21 points)
        if extracted.left_hand is not None and len(extracted.left_hand) >= 21:
            for j in range(21):
                pts[13 + j, 0] = float(extracted.left_hand[j, 0])
                pts[13 + j, 1] = float(extracted.left_hand[j, 1])

        # 3. Right hand (21 points)
        if extracted.right_hand is not None and len(extracted.right_hand) >= 21:
            for k in range(21):
                pts[34 + k, 0] = float(extracted.right_hand[k, 0])
                pts[34 + k, 1] = float(extracted.right_hand[k, 1])

        self.frame_buffer.append(pts)

    def reset(self) -> None:
        """Clear temporal frame buffer."""
        self.frame_buffer.clear()

    @torch.no_grad()
    def predict(self) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Run TGCN inference over temporal buffer.

        Returns:
            (best_gloss, best_confidence, top_3_predictions)
        """
        if len(self.frame_buffer) < 15:
            # Need minimum frames of motion to evaluate
            return None, 0.0, []

        buf_len = len(self.frame_buffer)
        raw_frames = np.stack(list(self.frame_buffer), axis=0)  # Shape (T, 55, 2)

        # Resample or pad to fixed length (num_samples = 50)
        if buf_len != self.num_samples:
            indices = np.linspace(0, buf_len - 1, self.num_samples).astype(np.int32)
            sampled = raw_frames[indices]  # Shape (50, 55, 2)
        else:
            sampled = raw_frames

        # Reshape to TGCN input format: (1, 55, 50 * 2 = 100)
        # sampled has shape (50, 55, 2) -> transpose to (55, 50, 2) -> reshape to (55, 100)
        transposed = np.transpose(sampled, (1, 0, 2))  # (55, 50, 2)
        flattened = transposed.reshape(55, self.num_samples * 2)  # (55, 100)

        tensor_in = torch.from_numpy(flattened).unsqueeze(0).float().to(self.device)  # (1, 55, 100)

        logits = self.model(tensor_in)  # (1, 100)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (100,)

        top_indices = np.argsort(probs)[::-1][:5]
        top_preds = [(self.vocab[idx], float(probs[idx])) for idx in top_indices]

        best_gloss, best_score = top_preds[0]
        if best_score >= self.min_confidence:
            return best_gloss, best_score, top_preds[:3]
        return None, best_score, top_preds[:3]
