"""Asset loading for the Torch Flappy renderer.

The CPU renderer uses the original Pygame surfaces directly.  Those sprites use
colorkeys rather than per-pixel alpha in the rgb-array path, so the renderer
stores RGB data plus a binary visibility mask for each sprite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pygame

from flappy_bird_gymnasium.envs import utils
from flappy_bird_gymnasium.envs.constants import (
    FILL_BACKGROUND_COLOR,
    PLAYER_ROT_THR,
)
from flappy_bird_gymnasium.envs.render_state import FlappyRenderState


@dataclass(frozen=True)
class Sprite:
    """RGB sprite data and its Pygame-colorkey visibility mask."""

    rgb: np.ndarray
    mask: np.ndarray
    name: str

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


def _surface_rgb(surface: pygame.Surface) -> np.ndarray:
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).astype(
        np.uint8, copy=True
    )


def _surface_mask(surface: pygame.Surface, rgb: np.ndarray) -> np.ndarray:
    colorkey = surface.get_colorkey()
    if colorkey is not None:
        key = np.asarray(colorkey[:3], dtype=np.uint8)
        return np.any(rgb != key, axis=2)

    flags = surface.get_flags()
    if flags & pygame.SRCALPHA:
        alpha = np.transpose(pygame.surfarray.array_alpha(surface), (1, 0))
        return alpha > 0

    return np.ones(rgb.shape[:2], dtype=bool)


def sprite_from_surface(surface: pygame.Surface, name: str) -> Sprite:
    rgb = _surface_rgb(surface)
    return Sprite(rgb=rgb, mask=_surface_mask(surface, rgb), name=name)


def common_visible_rotations() -> Tuple[int, ...]:
    """Rotations reachable from the stock env plus the capped reset/flap angle."""

    return tuple(range(-90, PLAYER_ROT_THR + 1))


@dataclass
class FlappyRenderAssets:
    """Pygame-derived sprites for one Flappy Bird asset variant."""

    bird_color: str = "yellow"
    pipe_color: str = "green"
    background: Optional[str] = "day"
    background_rgb: Optional[np.ndarray] = None
    fill_color: Tuple[int, int, int] = FILL_BACKGROUND_COLOR
    base: Optional[Sprite] = None
    pipe_upper: Optional[Sprite] = None
    pipe_lower: Optional[Sprite] = None
    birds: Optional[Dict[Tuple[int, int], Sprite]] = None

    @classmethod
    def load(
        cls,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: Optional[str] = "day",
        rotations: Optional[Iterable[int]] = None,
    ) -> "FlappyRenderAssets":
        images = utils.load_images(
            convert=False,
            bird_color=bird_color,
            pipe_color=pipe_color,
            bg_type=background,
        )

        background_rgb = None
        if images["background"] is not None:
            background_rgb = _surface_rgb(images["background"])

        rotation_values = common_visible_rotations() if rotations is None else rotations
        birds: Dict[Tuple[int, int], Sprite] = {}
        for player_idx, surface in enumerate(images["player"]):
            for rotation in rotation_values:
                angle = int(rotation)
                rotated = pygame.transform.rotate(surface, angle)
                birds[(player_idx, angle)] = sprite_from_surface(
                    rotated, name=f"bird-{player_idx}-rot-{angle}"
                )

        return cls(
            bird_color=bird_color,
            pipe_color=pipe_color,
            background=background,
            background_rgb=background_rgb,
            base=sprite_from_surface(images["base"], "base"),
            pipe_upper=sprite_from_surface(images["pipe"][0], "pipe-upper"),
            pipe_lower=sprite_from_surface(images["pipe"][1], "pipe-lower"),
            birds=birds,
        )

    @classmethod
    def for_state(cls, state: FlappyRenderState) -> "FlappyRenderAssets":
        return cls.load(
            bird_color=state.bird_color,
            pipe_color=state.pipe_color,
            background=state.background,
        )

    def bird_sprite(self, player_index: int, visible_rotation: int) -> Sprite:
        assert self.birds is not None
        key = (int(player_index), int(visible_rotation))
        try:
            return self.birds[key]
        except KeyError as exc:
            raise KeyError(
                f"missing bird sprite for index={player_index} "
                f"rotation={visible_rotation}"
            ) from exc

    def reference_frame(self, state: FlappyRenderState) -> np.ndarray:
        """Renders a numpy frame with the same draw order as the Torch renderer."""

        width, height = state.screen_size
        frame = np.empty((height, width, 3), dtype=np.uint8)
        if self.background_rgb is None:
            frame[:, :] = np.asarray(self.fill_color, dtype=np.uint8)
        else:
            frame[:, :] = self.background_rgb[:height, :width]

        assert self.pipe_upper is not None
        assert self.pipe_lower is not None
        assert self.base is not None

        for (upper_x, upper_y), (lower_x, lower_y) in zip(
            state.upper_pipes, state.lower_pipes
        ):
            _blit_numpy(frame, self.pipe_upper, int(upper_x), int(upper_y))
            _blit_numpy(frame, self.pipe_lower, int(lower_x), int(lower_y))
        _blit_numpy(frame, self.base, int(state.ground_x), int(state.ground_y))
        _blit_numpy(
            frame,
            self.bird_sprite(state.player_index, state.visible_rotation),
            int(state.player_x),
            int(state.player_y),
        )
        return frame


def _blit_numpy(frame: np.ndarray, sprite: Sprite, x: int, y: int) -> None:
    dst_h, dst_w = frame.shape[:2]
    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    width = min(sprite.width - src_x0, dst_w - dst_x0)
    height = min(sprite.height - src_y0, dst_h - dst_y0)
    if width <= 0 or height <= 0:
        return

    src_rgb = sprite.rgb[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
    src_mask = sprite.mask[src_y0 : src_y0 + height, src_x0 : src_x0 + width]
    dst = frame[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width]
    dst[src_mask] = src_rgb[src_mask]
