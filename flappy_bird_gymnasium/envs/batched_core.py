from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from gymnasium.utils import seeding

from flappy_bird_gymnasium.envs.constants import (
    BACKGROUND_WIDTH,
    BASE_WIDTH,
    PIPE_HEIGHT,
    PIPE_VEL_X,
    PIPE_WIDTH,
    PLAYER_ACC_Y,
    PLAYER_FLAP_ACC,
    PLAYER_HEIGHT,
    PLAYER_MAX_VEL_Y,
    PLAYER_ROT_THR,
    PLAYER_VEL_ROT,
    PLAYER_WIDTH,
)
from flappy_bird_gymnasium.envs.flappy_bird_env import Actions
from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState


@dataclass
class BatchedFlappyStepResult:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict[str, Any]]


class BatchedFlappyBirdCore:
    def __init__(
        self,
        batch_size: int,
        *,
        screen_size: tuple[int, int] = (288, 512),
        normalize_obs: bool = True,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: str | None = "day",
        score_limit: int | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.screen_width = int(screen_size[0])
        self.screen_height = int(screen_size[1])
        self.normalize_obs = bool(normalize_obs)
        self.pipe_gap = int(pipe_gap)
        self.bird_color = bird_color
        self.pipe_color = pipe_color
        self.background = background
        score_limit_value = -1 if score_limit is None else int(score_limit)
        self.score_limit = np.full(self.batch_size, score_limit_value, dtype=np.int64)

        self.player_x = np.full(self.batch_size, int(self.screen_width * 0.2), dtype=np.float64)
        self.player_y = np.zeros(self.batch_size, dtype=np.float64)
        self.player_vel_y = np.zeros(self.batch_size, dtype=np.float64)
        self.player_rot = np.zeros(self.batch_size, dtype=np.int64)
        self.player_idx = np.zeros(self.batch_size, dtype=np.int64)
        self._player_idx_cycle_pos = np.zeros(self.batch_size, dtype=np.int64)
        self.loop_iter = np.zeros(self.batch_size, dtype=np.int64)
        self.score = np.zeros(self.batch_size, dtype=np.int64)
        self.ground_x = np.zeros(self.batch_size, dtype=np.float64)
        self.ground_y = np.full(self.batch_size, self.screen_height * 0.79, dtype=np.float64)
        self.upper_pipe_xy = np.zeros((self.batch_size, 3, 2), dtype=np.float64)
        self.lower_pipe_xy = np.zeros((self.batch_size, 3, 2), dtype=np.float64)
        self.np_randoms = [seeding.np_random(None)[0] for _ in range(self.batch_size)]
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH

    def reset_batch(
        self,
        seeds: Sequence[int | None] | int | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        target_indices = self._indices(indices)
        seed_values = self._seed_values(seeds, len(target_indices))
        for offset, agent_idx in enumerate(target_indices):
            seed = seed_values[offset]
            if seed is not None:
                self.np_randoms[agent_idx] = seeding.np_random(int(seed))[0]
            self._reset_one(agent_idx)
        observations = self._observations()
        infos = [{"score": int(self.score[agent_idx])} for agent_idx in range(self.batch_size)]
        return observations, infos

    def step_batch(
        self,
        actions: Sequence[int] | np.ndarray,
        active_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> BatchedFlappyStepResult:
        action_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        active = np.ones(self.batch_size, dtype=np.bool_) if active_mask is None else np.asarray(active_mask, dtype=np.bool_).reshape(-1)

        rewards = np.full(self.batch_size, 0.1, dtype=np.float64)
        terminated = np.zeros(self.batch_size, dtype=np.bool_)
        truncated = np.zeros(self.batch_size, dtype=np.bool_)

        flap = active & (action_arr == int(Actions.FLAP)) & (self.player_y > -2 * PLAYER_HEIGHT)
        self.player_vel_y[flap] = PLAYER_FLAP_ACC

        player_mid_pos = self.player_x + PLAYER_WIDTH / 2
        scored = np.zeros(self.batch_size, dtype=np.bool_)
        for pipe_idx in range(3):
            pipe_mid_pos = self.upper_pipe_xy[:, pipe_idx, 0] + PIPE_WIDTH / 2
            passed = active & (pipe_mid_pos <= player_mid_pos) & (player_mid_pos < pipe_mid_pos + 4)
            self.score[passed] += 1
            scored |= passed
        rewards[scored] = 1.0

        animate = active & (((self.loop_iter + 1) % 3) == 0)
        cycle = np.asarray([0, 1, 2, 1], dtype=np.int64)
        self.player_idx[animate] = cycle[self._player_idx_cycle_pos[animate]]
        self._player_idx_cycle_pos[animate] = (self._player_idx_cycle_pos[animate] + 1) % 4

        self.loop_iter[active] = (self.loop_iter[active] + 1) % 30
        self.ground_x[active] = -((-self.ground_x[active] + 100) % self._base_shift)

        rotate = active & (self.player_rot > -90)
        self.player_rot[rotate] -= PLAYER_VEL_ROT

        gravity = active & ~flap & (self.player_vel_y < PLAYER_MAX_VEL_Y)
        self.player_vel_y[gravity] += PLAYER_ACC_Y
        self.player_rot[flap] = 45

        max_y_delta = self.ground_y - self.player_y - PLAYER_HEIGHT
        self.player_y[active] += np.minimum(self.player_vel_y, max_y_delta)[active]

        for pipe_idx in range(3):
            self.upper_pipe_xy[active, pipe_idx, 0] += PIPE_VEL_X
            self.lower_pipe_xy[active, pipe_idx, 0] += PIPE_VEL_X
            offscreen = active & (self.upper_pipe_xy[:, pipe_idx, 0] < -PIPE_WIDTH)
            for agent_idx in np.flatnonzero(offscreen):
                upper, lower = self._random_pipe(int(agent_idx))
                self.upper_pipe_xy[agent_idx, pipe_idx] = upper
                self.lower_pipe_xy[agent_idx, pipe_idx] = lower

        top_hit = active & (self.player_y < 0)
        rewards[top_hit] = -0.5

        crashed = active & self._check_crash()
        rewards[crashed] = -1.0
        terminated[crashed] = True
        self.player_vel_y[crashed] = 0

        limited = self.score_limit >= 0
        truncated[active & limited & (self.score >= self.score_limit)] = True

        rewards[~active] = 0.0
        observations = self._observations()
        infos = [{"score": int(self.score[agent_idx])} for agent_idx in range(self.batch_size)]
        return BatchedFlappyStepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    def render_state_batch(self) -> BatchedFlappyRenderState:
        visible_rotations = np.full(self.batch_size, PLAYER_ROT_THR, dtype=np.int16)
        mask = self.player_rot <= PLAYER_ROT_THR
        visible_rotations[mask] = self.player_rot[mask].astype(np.int16)
        return BatchedFlappyRenderState(
            player_xy=np.stack([self.player_x, self.player_y], axis=1).astype(np.float32),
            player_indices=self.player_idx.astype(np.int64),
            visible_rotations=visible_rotations,
            upper_pipe_xy=self.upper_pipe_xy.astype(np.float32),
            lower_pipe_xy=self.lower_pipe_xy.astype(np.float32),
            ground_xy=np.stack([self.ground_x, self.ground_y], axis=1).astype(np.float32),
            screen_width=int(self.screen_width),
            screen_height=int(self.screen_height),
            bird_color=self.bird_color,
            pipe_color=self.pipe_color,
            background=self.background,
        )

    def observations(self) -> np.ndarray:
        return self._observations()

    def _reset_one(self, agent_idx: int) -> None:
        self.player_x[agent_idx] = int(self.screen_width * 0.2)
        self.player_y[agent_idx] = int((self.screen_height - PLAYER_HEIGHT) / 2)
        self.player_vel_y[agent_idx] = -9
        self.player_rot[agent_idx] = 45
        self.player_idx[agent_idx] = 0
        self.loop_iter[agent_idx] = 0
        self.score[agent_idx] = 0
        self.ground_x[agent_idx] = 0
        self.ground_y[agent_idx] = self.screen_height * 0.79

        pipe1 = self._random_pipe(agent_idx)
        pipe2 = self._random_pipe(agent_idx)
        pipe3 = self._random_pipe(agent_idx)
        self.upper_pipe_xy[agent_idx] = np.asarray(
            [
                [self.screen_width, pipe1[0][1]],
                [self.screen_width + (self.screen_width / 2), pipe2[0][1]],
                [self.screen_width + self.screen_width, pipe3[0][1]],
            ],
            dtype=np.float64,
        )
        self.lower_pipe_xy[agent_idx] = np.asarray(
            [
                [self.screen_width, pipe1[1][1]],
                [self.screen_width + (self.screen_width / 2), pipe2[1][1]],
                [self.screen_width + self.screen_width, pipe3[1][1]],
            ],
            dtype=np.float64,
        )

    def _random_pipe(self, agent_idx: int) -> tuple[tuple[float, float], tuple[float, float]]:
        gap_ys = [20, 30, 40, 50, 60, 70, 80, 90]
        gap_y = gap_ys[int(self.np_randoms[agent_idx].integers(0, len(gap_ys)))]
        gap_y += int(self.ground_y[agent_idx] * 0.2)
        pipe_x = self.screen_width + PIPE_WIDTH + (self.screen_width * 0.2)
        return (
            (float(pipe_x), float(gap_y - PIPE_HEIGHT)),
            (float(pipe_x), float(gap_y + self.pipe_gap)),
        )

    def _observations(self) -> np.ndarray:
        pipes = np.empty((self.batch_size, 3, 3), dtype=np.float64)
        for pipe_idx in range(3):
            low_x = self.lower_pipe_xy[:, pipe_idx, 0]
            top_y = self.upper_pipe_xy[:, pipe_idx, 1] + PIPE_HEIGHT
            low_y = self.lower_pipe_xy[:, pipe_idx, 1]
            pipes[:, pipe_idx, 0] = low_x
            pipes[:, pipe_idx, 1] = top_y
            pipes[:, pipe_idx, 2] = low_y
            behind = low_x > self.screen_width
            pipes[behind, pipe_idx, 0] = self.screen_width
            pipes[behind, pipe_idx, 1] = 0
            pipes[behind, pipe_idx, 2] = self.screen_height

        order = np.argsort(pipes[:, :, 0], axis=1)
        sorted_pipes = np.take_along_axis(pipes, order[:, :, None], axis=1)
        pos_y = self.player_y.copy()
        vel_y = self.player_vel_y.copy()
        rot = self.player_rot.astype(np.float64)
        if self.normalize_obs:
            sorted_pipes[:, :, 0] /= self.screen_width
            sorted_pipes[:, :, 1] /= self.screen_height
            sorted_pipes[:, :, 2] /= self.screen_height
            pos_y /= self.screen_height
            vel_y /= PLAYER_MAX_VEL_Y
            rot /= 90

        return np.concatenate(
            [
                sorted_pipes.reshape(self.batch_size, 9),
                pos_y[:, None],
                vel_y[:, None],
                rot[:, None],
            ],
            axis=1,
        )

    def _check_crash(self) -> np.ndarray:
        ground = self.player_y + PLAYER_HEIGHT >= self.ground_y - 1
        crashed = ground.copy()
        candidates = ~ground
        player_x = self.player_x.astype(np.int64)
        player_y = self.player_y.astype(np.int64)
        for pipe_idx in range(3):
            up_x = self.upper_pipe_xy[:, pipe_idx, 0].astype(np.int64)
            up_y = self.upper_pipe_xy[:, pipe_idx, 1].astype(np.int64)
            low_x = self.lower_pipe_xy[:, pipe_idx, 0].astype(np.int64)
            low_y = self.lower_pipe_xy[:, pipe_idx, 1].astype(np.int64)
            up_collide = self._rects_collide(player_x, player_y, up_x, up_y)
            low_collide = self._rects_collide(player_x, player_y, low_x, low_y)
            crashed |= candidates & (up_collide | low_collide)
        return crashed

    @staticmethod
    def _rects_collide(player_x: np.ndarray, player_y: np.ndarray, pipe_x: np.ndarray, pipe_y: np.ndarray) -> np.ndarray:
        return (
            (player_x < pipe_x + PIPE_WIDTH)
            & (player_x + PLAYER_WIDTH > pipe_x)
            & (player_y < pipe_y + PIPE_HEIGHT)
            & (player_y + PLAYER_HEIGHT > pipe_y)
        )

    def _indices(self, indices: Sequence[int] | np.ndarray | None) -> list[int]:
        if indices is None:
            return list(range(self.batch_size))
        return [int(item) for item in np.asarray(indices).reshape(-1).tolist()]

    def _seed_values(self, seeds: Sequence[int | None] | int | None, count: int) -> list[int | None]:
        if seeds is None:
            return [None] * count
        if isinstance(seeds, (int, np.integer)):
            return [int(seeds)] * count
        return [None if item is None else int(item) for item in list(seeds)]
