"""Render-only state snapshots for Flappy Bird.

The classes in this module intentionally do not contain any game logic.  They
are a compact bridge between the CPU environment state and alternate renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


PipeState = Tuple[float, float]


@dataclass(frozen=True)
class FlappyRenderState:
    """Everything needed to draw one Flappy Bird frame without score or lidar."""

    player_x: float
    player_y: float
    player_index: int
    visible_rotation: int
    upper_pipes: Tuple[PipeState, ...]
    lower_pipes: Tuple[PipeState, ...]
    ground_x: float
    ground_y: float
    screen_width: int
    screen_height: int
    bird_color: str = "yellow"
    pipe_color: str = "green"
    background: Optional[str] = "day"

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self.screen_width, self.screen_height

    @property
    def pipe_count(self) -> int:
        return len(self.upper_pipes)


@dataclass(frozen=True)
class BatchedFlappyRenderState:
    """Numpy batch form consumed by vectorized renderers."""

    player_xy: np.ndarray
    player_indices: np.ndarray
    visible_rotations: np.ndarray
    upper_pipe_xy: np.ndarray
    lower_pipe_xy: np.ndarray
    ground_xy: np.ndarray
    screen_width: int
    screen_height: int
    bird_color: str = "yellow"
    pipe_color: str = "green"
    background: Optional[str] = "day"

    @classmethod
    def from_states(
        cls, states: Sequence[FlappyRenderState]
    ) -> "BatchedFlappyRenderState":
        if not states:
            raise ValueError("states must contain at least one render state")

        first = states[0]
        pipe_count = first.pipe_count
        for state in states:
            if state.screen_size != first.screen_size:
                raise ValueError("all render states must use the same screen size")
            if state.pipe_count != pipe_count:
                raise ValueError("all render states must have the same pipe count")
            if (
                state.bird_color != first.bird_color
                or state.pipe_color != first.pipe_color
                or state.background != first.background
            ):
                raise ValueError("batched render states must use one asset variant")

        return cls(
            player_xy=np.asarray(
                [(state.player_x, state.player_y) for state in states],
                dtype=np.float32,
            ),
            player_indices=np.asarray(
                [state.player_index for state in states], dtype=np.int64
            ),
            visible_rotations=np.asarray(
                [state.visible_rotation for state in states], dtype=np.int16
            ),
            upper_pipe_xy=np.asarray(
                [state.upper_pipes for state in states], dtype=np.float32
            ),
            lower_pipe_xy=np.asarray(
                [state.lower_pipes for state in states], dtype=np.float32
            ),
            ground_xy=np.asarray(
                [(state.ground_x, state.ground_y) for state in states],
                dtype=np.float32,
            ),
            screen_width=first.screen_width,
            screen_height=first.screen_height,
            bird_color=first.bird_color,
            pipe_color=first.pipe_color,
            background=first.background,
        )

    @classmethod
    def from_envs(cls, envs: Iterable[object]) -> "BatchedFlappyRenderState":
        states = []
        for env in envs:
            target = getattr(env, "unwrapped", env)
            states.append(target.get_render_state())
        return cls.from_states(states)

    @property
    def batch_size(self) -> int:
        return int(self.player_xy.shape[0])

    @property
    def pipe_count(self) -> int:
        return int(self.upper_pipe_xy.shape[1])

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self.screen_width, self.screen_height
