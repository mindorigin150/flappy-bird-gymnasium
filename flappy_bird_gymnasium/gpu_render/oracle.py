"""CPU image-observation oracle shared by parity tests and scripts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np
from PIL import Image


def resize_frame_pil(
    frame: np.ndarray, image_size: int = 84, grayscale: bool = False
) -> np.ndarray:
    """Matches ``RenderToImageWrapper`` in ``scripts/flappy_queue_latency_cnn.py``."""

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    image = Image.fromarray(arr)
    image = image.convert("L" if grayscale else "RGB")
    target_size = (int(image_size), int(image_size))
    if image.size != target_size:
        image = image.resize(target_size, Image.BILINEAR)

    out = np.asarray(image, dtype=np.uint8)
    if grayscale:
        out = out[..., None]
    return out


def latency_channel(
    height: int,
    width: int,
    frame_skip: int,
    max_frame_skip: int,
) -> np.ndarray:
    max_skip = max(1, int(max_frame_skip))
    skip_norm = min(int(frame_skip) / float(max_skip), 1.0)
    skip_value = int(round(skip_norm * 255.0))
    return np.full((height, width, 1), skip_value, dtype=np.uint8)


def append_latency_channel(
    obs: np.ndarray,
    frame_skip: int,
    max_frame_skip: int,
) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.uint8)
    plane = latency_channel(
        obs_arr.shape[0],
        obs_arr.shape[1],
        frame_skip=frame_skip,
        max_frame_skip=max_frame_skip,
    )
    return np.concatenate([obs_arr, plane], axis=2)


@dataclass
class FrameStackOracle:
    """Small HWC frame stacker mirroring the existing training wrapper order."""

    stack_frames: int = 4
    frames: Deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.stack_frames = max(1, int(self.stack_frames))
        self.frames = deque(maxlen=self.stack_frames)

    def reset(self, frame_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(frame_obs, dtype=np.uint8)
        self.frames.clear()
        for _ in range(self.stack_frames):
            self.frames.append(obs.copy())
        return self.current()

    def step(self, frame_obs: np.ndarray) -> np.ndarray:
        self.frames.append(np.asarray(frame_obs, dtype=np.uint8).copy())
        return self.current()

    def current(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("FrameStackOracle has not been reset")
        return np.concatenate(list(self.frames), axis=2)


@dataclass
class ImageObservationOracle:
    """Builds final CNN observations from raw RGB frames."""

    image_size: int = 84
    grayscale: bool = False
    stack_frames: int = 4
    obs_mode: str = "latency_channel"
    max_frame_skip: int = 1
    stacker: FrameStackOracle = field(init=False)

    def __post_init__(self) -> None:
        if self.obs_mode not in {"image_only", "latency_channel"}:
            raise ValueError(f"unsupported obs_mode: {self.obs_mode}")
        self.stacker = FrameStackOracle(self.stack_frames)

    def _frame_obs(self, frame: np.ndarray) -> np.ndarray:
        return resize_frame_pil(
            frame, image_size=self.image_size, grayscale=self.grayscale
        )

    def _maybe_latency(self, stacked: np.ndarray, frame_skip: int) -> np.ndarray:
        if self.obs_mode == "image_only":
            return stacked
        return append_latency_channel(
            stacked,
            frame_skip=frame_skip,
            max_frame_skip=self.max_frame_skip,
        )

    def reset(self, frame: np.ndarray, frame_skip: int = 0) -> np.ndarray:
        stacked = self.stacker.reset(self._frame_obs(frame))
        return self._maybe_latency(stacked, frame_skip)

    def step(self, frame: np.ndarray, frame_skip: int = 0) -> np.ndarray:
        stacked = self.stacker.step(self._frame_obs(frame))
        return self._maybe_latency(stacked, frame_skip)


def build_observation(
    frame: np.ndarray,
    image_size: int = 84,
    grayscale: bool = False,
    frame_skip: Optional[int] = None,
    max_frame_skip: int = 1,
) -> np.ndarray:
    obs = resize_frame_pil(frame, image_size=image_size, grayscale=grayscale)
    if frame_skip is None:
        return obs
    return append_latency_channel(obs, frame_skip, max_frame_skip)
