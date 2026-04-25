"""Torch prototype renderer for Flappy Bird frames."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from flappy_bird_gymnasium.envs.render_state import (
    BatchedFlappyRenderState,
    FlappyRenderState,
)
from flappy_bird_gymnasium.gpu_render.assets import FlappyRenderAssets, Sprite
from flappy_bird_gymnasium.gpu_render.oracle import resize_frame_pil


StateLike = Union[
    FlappyRenderState,
    BatchedFlappyRenderState,
    Iterable[FlappyRenderState],
]


@dataclass(frozen=True)
class TorchSprite:
    rgb: torch.Tensor
    mask: torch.Tensor
    name: str

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass(frozen=True)
class DirectTorchSprite:
    premul_rgb: torch.Tensor
    alpha: torch.Tensor
    name: str

    @property
    def height(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def width(self) -> int:
        return int(self.alpha.shape[1])


class TorchFlappyRenderer:
    """Binary-colorkey Torch blitter matching the env's ``rgb_array`` draw path."""

    def __init__(
        self,
        assets: FlappyRenderAssets,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.assets = assets
        self.device = torch.device(device or "cpu")
        self._background = (
            None
            if assets.background_rgb is None
            else torch.as_tensor(
                assets.background_rgb, dtype=torch.uint8, device=self.device
            )
        )
        self._fill_color = torch.tensor(
            assets.fill_color, dtype=torch.uint8, device=self.device
        )
        assert assets.base is not None
        assert assets.pipe_upper is not None
        assert assets.pipe_lower is not None
        assert assets.birds is not None
        self._base = self._to_torch_sprite(assets.base)
        self._pipe_upper = self._to_torch_sprite(assets.pipe_upper)
        self._pipe_lower = self._to_torch_sprite(assets.pipe_lower)
        self._birds: Dict[Tuple[int, int], TorchSprite] = {
            key: self._to_torch_sprite(sprite) for key, sprite in assets.birds.items()
        }
        self._direct_background_cache: Dict[int, torch.Tensor] = {}
        self._direct_sprite_cache: Dict[Tuple[str, int, int], DirectTorchSprite] = {}
        self._direct_grid_cache: Dict[
            Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _to_torch_sprite(self, sprite: Sprite) -> TorchSprite:
        return TorchSprite(
            rgb=torch.as_tensor(sprite.rgb, dtype=torch.uint8, device=self.device),
            mask=torch.as_tensor(sprite.mask, dtype=torch.bool, device=self.device),
            name=sprite.name,
        )

    def _batch(self, states: StateLike) -> BatchedFlappyRenderState:
        if isinstance(states, BatchedFlappyRenderState):
            return states
        if isinstance(states, FlappyRenderState):
            return BatchedFlappyRenderState.from_states([states])
        return BatchedFlappyRenderState.from_states(list(states))

    def render(self, states: StateLike) -> torch.Tensor:
        """Renders full-size frames as ``uint8`` HWC tensors.

        Returns:
            Tensor shaped ``(batch, screen_height, screen_width, 3)``.
        """

        batch = self._batch(states)
        self._validate_assets(batch)
        frame = self._background_frame(batch)

        for env_idx in range(batch.batch_size):
            for pipe_idx in range(batch.pipe_count):
                x, y = batch.upper_pipe_xy[env_idx, pipe_idx]
                self._blit(frame[env_idx], self._pipe_upper, int(x), int(y))
                x, y = batch.lower_pipe_xy[env_idx, pipe_idx]
                self._blit(frame[env_idx], self._pipe_lower, int(x), int(y))

            ground_x, ground_y = batch.ground_xy[env_idx]
            self._blit(frame[env_idx], self._base, int(ground_x), int(ground_y))

            player_index = int(batch.player_indices[env_idx])
            rotation = int(batch.visible_rotations[env_idx])
            player = self._birds[(player_index, rotation)]
            player_x, player_y = batch.player_xy[env_idx]
            self._blit(frame[env_idx], player, int(player_x), int(player_y))

        return frame

    def render_numpy(self, states: StateLike) -> np.ndarray:
        rendered = self.render(states).detach().cpu().numpy()
        if isinstance(states, FlappyRenderState):
            return rendered[0]
        return rendered

    def render_observation(
        self,
        states: StateLike,
        image_size: int = 84,
        grayscale: bool = False,
        resize_backend: str = "pil",
        profiler=None,
    ) -> torch.Tensor:
        """Renders resized HWC observations.

        ``resize_backend="pil"`` is the pixel-exact oracle path used by tests.
        ``resize_backend="torch"`` keeps resizing on the tensor device but can
        differ from Pillow by a small rounding amount on downsampling.
        """

        render_timer = (
            profiler.time("full_render_s") if profiler is not None else nullcontext()
        )
        with render_timer:
            frames = self.render(states)
        if resize_backend == "pil":
            resize_timer = (
                profiler.time("resize_s") if profiler is not None else nullcontext()
            )
            with resize_timer:
                obs = [
                    resize_frame_pil(frame, image_size=image_size, grayscale=grayscale)
                    for frame in frames.detach().cpu().numpy()
                ]
                return torch.as_tensor(np.stack(obs, axis=0), device=self.device)
        if resize_backend != "torch":
            raise ValueError(f"unsupported resize_backend: {resize_backend}")

        resize_timer = (
            profiler.time("resize_s") if profiler is not None else nullcontext()
        )
        with resize_timer:
            nchw = frames.permute(0, 3, 1, 2).float()
            if grayscale:
                r = nchw[:, 0:1]
                g = nchw[:, 1:2]
                b = nchw[:, 2:3]
                nchw = torch.floor(
                    (r * 299.0 + g * 587.0 + b * 114.0 + 500.0) / 1000.0
                )
            resized = F.interpolate(
                nchw,
                size=(int(image_size), int(image_size)),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            return resized.round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

    def render_observation_direct(
        self,
        states: StateLike,
        image_size: int = 84,
        grayscale: bool = False,
        profiler=None,
    ) -> torch.Tensor:
        """Renders training observations directly at ``image_size``.

        This path intentionally skips the full-size RGB frame.  It is optimized
        for PPO/CNN observations and is therefore an approximation of
        ``render() -> torch resize`` rather than a replacement for
        ``rgb_array`` visualization.
        """

        timer = profiler.time("direct_obs_s") if profiler is not None else nullcontext()
        with timer:
            batch = self._batch(states)
            self._validate_assets(batch)
            image_size = int(image_size)
            if image_size <= 0:
                raise ValueError("image_size must be positive")

            frame = self._direct_background_frame(batch, image_size)
            x_scale = image_size / float(batch.screen_width)
            y_scale = image_size / float(batch.screen_height)

            for pipe_idx in range(batch.pipe_count):
                self._direct_blit_scaled_batch(
                    frame,
                    self._pipe_upper,
                    batch.upper_pipe_xy[:, pipe_idx],
                    x_scale,
                    y_scale,
                )
                self._direct_blit_scaled_batch(
                    frame,
                    self._pipe_lower,
                    batch.lower_pipe_xy[:, pipe_idx],
                    x_scale,
                    y_scale,
                )

            self._direct_blit_scaled_batch(
                frame,
                self._base,
                batch.ground_xy,
                x_scale,
                y_scale,
            )

            for player_index, rotation in sorted(
                set(
                    zip(
                        batch.player_indices.astype(np.int64, copy=False).tolist(),
                        batch.visible_rotations.astype(np.int64, copy=False).tolist(),
                    )
                )
            ):
                env_indices = np.flatnonzero(
                    (batch.player_indices == player_index)
                    & (batch.visible_rotations == rotation)
                )
                if env_indices.size == 0:
                    continue
                self._direct_blit_scaled_batch(
                    frame,
                    self._birds[(int(player_index), int(rotation))],
                    batch.player_xy[env_indices],
                    x_scale,
                    y_scale,
                    frame_indices=env_indices,
                )

            if grayscale:
                r = frame[:, :, :, 0:1]
                g = frame[:, :, :, 1:2]
                b = frame[:, :, :, 2:3]
                frame = torch.floor(
                    (r * 299.0 + g * 587.0 + b * 114.0 + 500.0) / 1000.0
                )
                return frame.clamp_(0, 255).to(torch.uint8)

            return frame.round().clamp_(0, 255).to(torch.uint8)

    def _render_observation_direct_scalar(
        self,
        states: StateLike,
        image_size: int = 84,
        grayscale: bool = False,
    ) -> torch.Tensor:
        """Reference direct observation path used by parity tests."""

        batch = self._batch(states)
        self._validate_assets(batch)
        image_size = int(image_size)
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        frame = self._direct_background_frame(batch, image_size)
        x_scale = image_size / float(batch.screen_width)
        y_scale = image_size / float(batch.screen_height)

        for env_idx in range(batch.batch_size):
            for pipe_idx in range(batch.pipe_count):
                x, y = batch.upper_pipe_xy[env_idx, pipe_idx]
                self._direct_blit_scaled(
                    frame[env_idx],
                    self._pipe_upper,
                    int(x),
                    int(y),
                    x_scale,
                    y_scale,
                )
                x, y = batch.lower_pipe_xy[env_idx, pipe_idx]
                self._direct_blit_scaled(
                    frame[env_idx],
                    self._pipe_lower,
                    int(x),
                    int(y),
                    x_scale,
                    y_scale,
                )

            ground_x, ground_y = batch.ground_xy[env_idx]
            self._direct_blit_scaled(
                frame[env_idx],
                self._base,
                int(ground_x),
                int(ground_y),
                x_scale,
                y_scale,
            )

            player_index = int(batch.player_indices[env_idx])
            rotation = int(batch.visible_rotations[env_idx])
            player = self._birds[(player_index, rotation)]
            player_x, player_y = batch.player_xy[env_idx]
            self._direct_blit_scaled(
                frame[env_idx],
                player,
                int(player_x),
                int(player_y),
                x_scale,
                y_scale,
            )

        if grayscale:
            r = frame[:, :, :, 0:1]
            g = frame[:, :, :, 1:2]
            b = frame[:, :, :, 2:3]
            frame = torch.floor(
                (r * 299.0 + g * 587.0 + b * 114.0 + 500.0) / 1000.0
            )
            return frame.clamp_(0, 255).to(torch.uint8)

        return frame.round().clamp_(0, 255).to(torch.uint8)

    def _validate_assets(self, batch: BatchedFlappyRenderState) -> None:
        if batch.bird_color != self.assets.bird_color:
            raise ValueError(
                f"renderer bird_color={self.assets.bird_color!r} cannot draw "
                f"state bird_color={batch.bird_color!r}"
            )
        if batch.pipe_color != self.assets.pipe_color:
            raise ValueError(
                f"renderer pipe_color={self.assets.pipe_color!r} cannot draw "
                f"state pipe_color={batch.pipe_color!r}"
            )
        if batch.background != self.assets.background:
            raise ValueError(
                f"renderer background={self.assets.background!r} cannot draw "
                f"state background={batch.background!r}"
            )

    def _background_frame(self, batch: BatchedFlappyRenderState) -> torch.Tensor:
        height = int(batch.screen_height)
        width = int(batch.screen_width)
        if self._background is None:
            frame = torch.empty(
                (batch.batch_size, height, width, 3),
                dtype=torch.uint8,
                device=self.device,
            )
            frame[:, :, :, :] = self._fill_color
            return frame

        bg = self._background[:height, :width]
        return bg.unsqueeze(0).expand(batch.batch_size, -1, -1, -1).clone()

    def _direct_background_frame(
        self,
        batch: BatchedFlappyRenderState,
        image_size: int,
    ) -> torch.Tensor:
        if self._background is None:
            frame = torch.empty(
                (batch.batch_size, image_size, image_size, 3),
                dtype=torch.float32,
                device=self.device,
            )
            frame[:, :, :, :] = self._fill_color.float()
            return frame

        background = self._direct_background_cache.get(image_size)
        if background is None:
            nchw = self._background.permute(2, 0, 1).unsqueeze(0).float()
            background = F.interpolate(
                nchw,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0].permute(1, 2, 0)
            self._direct_background_cache[image_size] = background
        return background.unsqueeze(0).expand(batch.batch_size, -1, -1, -1).clone()

    def _blit(self, dst: torch.Tensor, sprite: TorchSprite, x: int, y: int) -> None:
        dst_h = int(dst.shape[0])
        dst_w = int(dst.shape[1])
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
        patch = dst[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width]
        patch[src_mask] = src_rgb[src_mask]

    def _scaled_sprite(
        self,
        sprite: TorchSprite,
        width: int,
        height: int,
    ) -> DirectTorchSprite:
        key = (sprite.name, int(width), int(height))
        cached = self._direct_sprite_cache.get(key)
        if cached is not None:
            return cached

        mask = sprite.mask.float().unsqueeze(0).unsqueeze(0)
        rgb = sprite.rgb.float().permute(2, 0, 1).unsqueeze(0)
        premul = rgb * mask

        premul_resized = F.interpolate(
            premul,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].permute(1, 2, 0)
        alpha_resized = F.interpolate(
            mask,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].permute(1, 2, 0).clamp_(0.0, 1.0)

        scaled = DirectTorchSprite(
            premul_rgb=premul_resized,
            alpha=alpha_resized,
            name=sprite.name,
        )
        self._direct_sprite_cache[key] = scaled
        return scaled

    def _direct_blit_scaled(
        self,
        dst: torch.Tensor,
        sprite: TorchSprite,
        source_x: int,
        source_y: int,
        x_scale: float,
        y_scale: float,
    ) -> None:
        width = max(1, _round_half_up(sprite.width * x_scale))
        height = max(1, _round_half_up(sprite.height * y_scale))
        scaled = self._scaled_sprite(sprite, width=width, height=height)
        x = _round_half_up(source_x * x_scale)
        y = _round_half_up(source_y * y_scale)

        dst_h = int(dst.shape[0])
        dst_w = int(dst.shape[1])
        src_x0 = max(0, -x)
        src_y0 = max(0, -y)
        dst_x0 = max(0, x)
        dst_y0 = max(0, y)
        clipped_width = min(scaled.width - src_x0, dst_w - dst_x0)
        clipped_height = min(scaled.height - src_y0, dst_h - dst_y0)
        if clipped_width <= 0 or clipped_height <= 0:
            return

        src_premul = scaled.premul_rgb[
            src_y0 : src_y0 + clipped_height,
            src_x0 : src_x0 + clipped_width,
        ]
        src_alpha = scaled.alpha[
            src_y0 : src_y0 + clipped_height,
            src_x0 : src_x0 + clipped_width,
        ]
        patch = dst[
            dst_y0 : dst_y0 + clipped_height,
            dst_x0 : dst_x0 + clipped_width,
        ]
        patch.mul_(1.0 - src_alpha).add_(src_premul)

    def _direct_blit_scaled_batch(
        self,
        dst: torch.Tensor,
        sprite: TorchSprite,
        source_xy: np.ndarray,
        x_scale: float,
        y_scale: float,
        frame_indices: Optional[np.ndarray] = None,
    ) -> None:
        width = max(1, _round_half_up(sprite.width * x_scale))
        height = max(1, _round_half_up(sprite.height * y_scale))
        scaled = self._scaled_sprite(sprite, width=width, height=height)

        positions = np.asarray(source_xy)
        if positions.size == 0:
            return
        positions = positions.reshape(-1, 2)
        if frame_indices is None:
            frame_indices_np = np.arange(positions.shape[0], dtype=np.int64)
        else:
            frame_indices_np = np.asarray(frame_indices, dtype=np.int64)
        if frame_indices_np.size != positions.shape[0]:
            raise ValueError("frame_indices must match source_xy length")

        source_x = positions[:, 0].astype(np.int64, copy=False)
        source_y = positions[:, 1].astype(np.int64, copy=False)
        x = np.floor(source_x.astype(np.float64) * float(x_scale) + 0.5).astype(
            np.int64
        )
        y = np.floor(source_y.astype(np.float64) * float(y_scale) + 0.5).astype(
            np.int64
        )

        dst_h = int(dst.shape[1])
        dst_w = int(dst.shape[2])
        src_x0 = np.maximum(0, -x)
        src_y0 = np.maximum(0, -y)
        dst_x0 = np.maximum(0, x)
        dst_y0 = np.maximum(0, y)
        clipped_width = np.minimum(scaled.width - src_x0, dst_w - dst_x0)
        clipped_height = np.minimum(scaled.height - src_y0, dst_h - dst_y0)
        visible = (clipped_width > 0) & (clipped_height > 0)
        if not bool(np.any(visible)):
            return

        frame_indices_np = frame_indices_np[visible]
        x = x[visible]
        y = y[visible]

        n = int(frame_indices_np.shape[0])
        grid_y, grid_x = self._direct_sprite_grid(scaled.height, scaled.width)
        grid_y = grid_y.unsqueeze(0).expand(n, -1, -1)
        grid_x = grid_x.unsqueeze(0).expand(n, -1, -1)

        x_t = torch.as_tensor(x, dtype=torch.long, device=self.device).view(n, 1, 1)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device).view(n, 1, 1)
        dst_x = x_t + grid_x
        dst_y = y_t + grid_y
        in_bounds = (
            (dst_x >= 0)
            & (dst_x < dst_w)
            & (dst_y >= 0)
            & (dst_y < dst_h)
        )

        env_t = torch.as_tensor(
            frame_indices_np, dtype=torch.long, device=self.device
        ).view(n, 1, 1)
        env_idx = env_t.expand_as(in_bounds)[in_bounds]
        dst_y_idx = dst_y[in_bounds]
        dst_x_idx = dst_x[in_bounds]

        alpha = scaled.alpha[:, :, 0].unsqueeze(0).expand(n, -1, -1)[in_bounds]
        premul = scaled.premul_rgb.unsqueeze(0).expand(n, -1, -1, -1)[in_bounds]
        current = dst[env_idx, dst_y_idx, dst_x_idx]
        dst[env_idx, dst_y_idx, dst_x_idx] = (
            current * (1.0 - alpha.unsqueeze(1)) + premul
        )

    def _direct_sprite_grid(
        self, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (int(height), int(width))
        cached = self._direct_grid_cache.get(key)
        if cached is not None:
            return cached

        y = torch.arange(int(height), dtype=torch.long, device=self.device)
        x = torch.arange(int(width), dtype=torch.long, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self._direct_grid_cache[key] = (grid_y, grid_x)
        return grid_y, grid_x


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))
