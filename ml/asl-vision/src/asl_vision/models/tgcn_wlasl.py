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

# 13 MediaPipe upper-body pose joints used in 55-keypoint skeletal layout.
# NOTE: the 55-node TGCN checkpoint was trained on OpenPose BODY_25 order
# (kept joints): Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow,
# LWrist, MidHip, REye, LEye, REar, LEar. MediaPipe-33 has no Neck/MidHip
# landmarks, so those slots are synthesized as shoulder/hip midpoints.
# Do NOT reorder this tuple: slot order must match the training layout.
# Maps to MediaPipe-33 indices below via _extract_upper_body_pose().
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

# MediaPipe-33 indices ordered to reproduce the BODY_25 training slot order.
# Slots 1 (Neck) and 8 (MidHip) are None: synthesized as midpoints.
_BODY25_MP_ORDER: tuple[int | None, ...] = (
    0,     # Nose
    None,  # Neck = midpoint(11, 12)
    12,    # Right shoulder
    14,    # Right elbow
    16,    # Right wrist
    11,    # Left shoulder
    13,    # Left elbow
    15,    # Left wrist
    None,  # MidHip = midpoint(23, 24)
    4,     # Right eye inner
    1,     # Left eye inner
    8,     # Right ear
    7,     # Left ear
)

_MIDPOINT_PAIRS: dict[int, tuple[int, int]] = {1: (11, 12), 8: (23, 24)}


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
        if x.dim() == 4:
            # Adapt packed spatiotemporal tensor (B, C, T, V) to (B, N=55, F=100).
            # Pose slots must reproduce the BODY_25 training order; Neck and
            # MidHip are synthesized as shoulder/hip midpoints (see _BODY25_MP_ORDER).
            B, _, _, V = x.shape
            if V >= 75:
                c2 = x[:, :2, :, :]
                nose = c2[:, :, :, [0]]
                neck = (c2[:, :, :, [11]] + c2[:, :, :, [12]]) / 2.0
                rsh = c2[:, :, :, [12]]
                relb = c2[:, :, :, [14]]
                rwri = c2[:, :, :, [16]]
                lsh = c2[:, :, :, [11]]
                lelb = c2[:, :, :, [13]]
                lwri = c2[:, :, :, [15]]
                midhip = (c2[:, :, :, [23]] + c2[:, :, :, [24]]) / 2.0
                reye = c2[:, :, :, [4]]
                leye = c2[:, :, :, [1]]
                rear = c2[:, :, :, [8]]
                lear = c2[:, :, :, [7]]
                pose13 = torch.cat(
                    [nose, neck, rsh, relb, rwri, lsh, lelb, lwri, midhip, reye, leye, rear, lear],
                    dim=-1,
                )
                hands = x[:, :2, :, 33:75]
                sub = torch.cat([pose13, hands], dim=-1)
            elif V == 55:
                sub = x[:, :2, :, :]
            else:
                raise ValueError(f"Expected 55 or >=75 joints for TGCN adaptation, got {V}.")
            _, c_sub, t_sub, v_sub = sub.shape
            sub_reshaped = sub.permute(0, 1, 3, 2).reshape(B, c_sub * v_sub, t_sub)
            if t_sub != 50:
                resampled = torch.nn.functional.interpolate(
                    sub_reshaped, size=50, mode="linear", align_corners=False
                )
            else:
                resampled = sub_reshaped
            resampled = resampled.view(B, c_sub, v_sub, 50).permute(0, 2, 3, 1)
            x = resampled.reshape(B, v_sub, 100)

        b, n, _ = x.shape
        y = self.gc1(x)
        y = self.bn1(y.view(b, -1)).view(b, n, -1)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = torch.mean(y, dim=1)
        return self.fc_out(out)


# Alias matching architecture naming conventions
TGCNModel = GCN_muti_att


def _extract_upper_body_pose(
    pose: np.ndarray | None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> np.ndarray:
    """Extract 13 upper-body pose joints in BODY_25 training slot order.

    Replicates the WLASL training normalization ``2 * (px / 256 - 0.5)`` where
    ``px`` are native pixel coordinates. MediaPipe yields normalized [0, 1]
    coordinates, so pixels are recovered as ``norm * frame_dim``. Defaults of
    256 preserve the legacy ``2 * (norm - 0.5)`` behavior when dims are unknown.
    """
    out = np.full((13, 2), -1.0, dtype=np.float32)
    if pose is None or len(pose) == 0:
        return out

    w = float(frame_width) if frame_width else 256.0
    h = float(frame_height) if frame_height else 256.0

    def _norm(p: np.ndarray) -> np.ndarray | None:
        if np.isnan(p[0]) or np.isnan(p[1]) or (float(p[0]) == 0.0 and float(p[1]) == 0.0):
            return None
        x = 2.0 * ((float(p[0]) * w) / 256.0 - 0.5)
        y = 2.0 * ((float(p[1]) * h) / 256.0 - 0.5)
        return np.array([x, y], dtype=np.float32)

    n_pose = len(pose)
    cache: dict[int, np.ndarray | None] = {}
    for slot, mp_idx in enumerate(_BODY25_MP_ORDER):
        if mp_idx is not None:
            if mp_idx not in cache:
                cache[mp_idx] = _norm(pose[mp_idx]) if n_pose > mp_idx else None
            val = cache[mp_idx]
        else:
            pair = _MIDPOINT_PAIRS[slot]
            a = cache.get(pair[0])
            if pair[0] not in cache:
                a = _norm(pose[pair[0]]) if n_pose > pair[0] else None
                cache[pair[0]] = a
            b = cache.get(pair[1])
            if pair[1] not in cache:
                b = _norm(pose[pair[1]]) if n_pose > pair[1] else None
                cache[pair[1]] = b
            val = (a + b) / 2.0 if (a is not None and b is not None) else None
        if val is not None:
            out[slot] = val

    return out


class TGCNWLASLClassifier:
    """Pretrained WLASL-100 Temporal Graph Convolutional Network inference engine."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        num_samples: int = 50,
        min_confidence: float = 0.45,
        min_active_frames: int = 15,
        min_motion: float = 0.020,
        device: str = "cpu",
    ) -> None:
        self.num_samples = num_samples
        self.min_confidence = min_confidence
        self.min_active_frames = min_active_frames
        self.min_motion = min_motion
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
        self.active_hand_frames: int = 0
        self.idle_frames: int = 0

    def add_frame(self, extracted: Any) -> None:
        """Extract 55 keypoints (13 upper pose + 21 left hand + 21 right hand) and append to buffer."""
        has_left = extracted.left_hand is not None and len(extracted.left_hand) >= 21
        has_right = extracted.right_hand is not None and len(extracted.right_hand) >= 21

        # Check if hands are active in signing space (reject resting hands below desk/lap)
        left_active = has_left and (float(extracted.left_hand[0, 1]) < 0.76 or float(extracted.left_hand[8, 1]) < 0.72)
        right_active = has_right and (float(extracted.right_hand[0, 1]) < 0.76 or float(extracted.right_hand[8, 1]) < 0.72)
        has_active_hand = left_active or right_active

        # Training-scale pixel recovery: WLASL normalization is 2*(px/256-0.5).
        fw = float(getattr(extracted, "frame_width", None) or 256)
        fh = float(getattr(extracted, "frame_height", None) or 256)

        def _scale_hand(hand: np.ndarray) -> np.ndarray:
            px = hand[:, :2].astype(np.float32).copy()
            px[:, 0] = 2.0 * ((px[:, 0] * fw) / 256.0 - 0.5)
            px[:, 1] = 2.0 * ((px[:, 1] * fh) / 256.0 - 0.5)
            return px

        if not has_active_hand:
            self.idle_frames += 1
            if self.idle_frames >= 10:
                self.reset()
                return
            # If an active stroke was underway, allow brief occlusion/transition up to 9 frames
            # Preserve pose coordinates so temporal graph convolution continuity is maintained
            if self.active_hand_frames > 0:
                pts = np.full((55, 2), -1.0, dtype=np.float32)
                pts[:13] = _extract_upper_body_pose(
                    extracted.pose, frame_width=fw, frame_height=fh
                )
                self.frame_buffer.append(pts)
            return

        self.idle_frames = 0
        self.active_hand_frames += 1

        # Frame shape: (55, 2), unobserved joints initialize to -1.0
        pts = np.full((55, 2), -1.0, dtype=np.float32)

        # 1. Upper-body pose (13 points mapped to OpenPose BODY_25 layout)
        pts[:13] = _extract_upper_body_pose(
            extracted.pose, frame_width=fw, frame_height=fh
        )

        # 2. Left hand (21 points)
        if left_active:
            pts[13:34] = _scale_hand(extracted.left_hand)

        # 3. Right hand (21 points)
        if right_active:
            pts[34:55] = _scale_hand(extracted.right_hand)

        self.frame_buffer.append(pts)

    def reset(self) -> None:
        """Clear temporal frame buffer and stroke tracking state."""
        self.frame_buffer.clear()
        self.active_hand_frames = 0
        self.idle_frames = 0

    @torch.no_grad()
    def predict(self) -> tuple[str | None, float, list[tuple[str, float]]]:
        """Run TGCN inference over temporal buffer.

        Returns:
            (best_gloss, best_confidence, top_3_predictions)
        """
        if len(self.frame_buffer) < self.min_active_frames or self.active_hand_frames < self.min_active_frames:
            # Need minimum frames of motion to evaluate
            return None, 0.0, []

        buf_len = len(self.frame_buffer)
        raw_frames = np.stack(list(self.frame_buffer), axis=0)  # Shape (T, 55, 2)

        # Dynamic temporal motion check: verify active hand has kinematic displacement over time
        # (reject static resting posture or frozen hands)
        lh_pts = raw_frames[:, 13:34, :]  # (T, 21, 2)
        rh_pts = raw_frames[:, 34:55, :]  # (T, 21, 2)

        def _calc_temporal_motion(hand_pts: np.ndarray) -> float:
            valid_mask = (hand_pts > -0.99).all(axis=-1)
            joint_stds: list[float] = []
            for j in range(21):
                v = valid_mask[:, j]
                if v.sum() >= 10:
                    j_std = float(np.std(hand_pts[v, j, :], axis=0).mean())
                    joint_stds.append(j_std)
            if not joint_stds:
                return 0.0
            return float(np.mean(joint_stds))

        lh_motion = _calc_temporal_motion(lh_pts)
        rh_motion = _calc_temporal_motion(rh_pts)

        max_motion = max(lh_motion, rh_motion)
        if max_motion < self.min_motion:
            # Insufficient dynamic motion (hands held static or resting)
            return None, 0.0, []

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
