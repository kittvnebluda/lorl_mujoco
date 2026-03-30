from copy import copy
from enum import Enum, auto
from importlib.resources import files

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray

from ..utils import quat2euler

DUMMY_ACTION = np.zeros(12, dtype=np.float32)


class HealthState(Enum):
    ALIVE = auto()
    TERMINATED = auto()
    TRUNCATED = auto()

    @property
    def terminated(self):
        return self is HealthState.TERMINATED

    @property
    def truncated(self):
        return self is HealthState.TRUNCATED


class UnitreeGo2Env(MujocoEnv, RecordConstructorArgs):
    """
    +-----+-----------------------------------------------+---------+---------+------------------+-------+----------------+
    | Idx | Description                                   | Min     | Max     | Name             | Joint | Unit           |
    +-----+-----------------------------------------------+---------+---------+------------------+-------+----------------+
    | qpos (positions/configuration, length 19)                                                                           |
    +-----+-----------------------------------------------+---------+---------+------------------+-------+----------------+
    |  0  | x-coordinate of the base (torso)              | -Inf    | Inf     | base (freejoint) | free  | position (m)   |
    |  1  | y-coordinate of the base (torso)              | -Inf    | Inf     | base (freejoint) | free  | position (m)   |
    |  2  | z-coordinate of the base (torso)              | -Inf    | Inf     | base (freejoint) | free  | position (m)   |
    |  3  | w component of base orientation quaternion    | -1      | 1       | base (freejoint) | free  | quaternion     |
    |  4  | x component of base orientation quaternion    | -1      | 1       | base (freejoint) | free  | quaternion     |
    |  5  | y component of base orientation quaternion    | -1      | 1       | base (freejoint) | free  | quaternion     |
    |  6  | z component of base orientation quaternion    | -1      | 1       | base (freejoint) | free  | quaternion     |
    |  7  | FL abduction / hip roll angle                 | -1.0472 | 1.0472  | FL_hip_joint     | hinge | angle (rad)    |
    |  8  | FL thigh / hip pitch angle                    | -1.5708 | 3.4907  | FL_thigh_joint   | hinge | angle (rad)    |
    |  9  | FL calf / knee pitch angle                    | -2.7227 | -0.8378 | FL_calf_joint    | hinge | angle (rad)    |
    | 10  | FR abduction / hip roll angle                 | -1.0472 | 1.0472  | FR_hip_joint     | hinge | angle (rad)    |
    | 11  | FR thigh / hip pitch angle                    | -1.5708 | 3.4907  | FR_thigh_joint   | hinge | angle (rad)    |
    | 12  | FR calf / knee pitch angle                    | -2.7227 | -0.8378 | FR_calf_joint    | hinge | angle (rad)    |
    | 13  | RL abduction / hip roll angle                 | -1.0472 | 1.0472  | RL_hip_joint     | hinge | angle (rad)    |
    | 14  | RL thigh / hip pitch angle                    | -0.5236 | 4.5379  | RL_thigh_joint   | hinge | angle (rad)    |
    | 15  | RL calf / knee pitch angle                    | -2.7227 | -0.8378 | RL_calf_joint    | hinge | angle (rad)    |
    | 16  | RR abduction / hip roll angle                 | -1.0472 | 1.0472  | RR_hip_joint     | hinge | angle (rad)    |
    | 17  | RR thigh / hip pitch angle                    | -0.5236 | 4.5379  | RR_thigh_joint   | hinge | angle (rad)    |
    | 18  | RR calf / knee pitch angle                    | -2.7227 | -0.8378 | RR_calf_joint    | hinge | angle (rad)    |
    +-----+-----------------------------------------------+---------+---------+------------------+-------+----------------+
    | qvel (velocities, length 18)                                                                                        |
    +-----+-----------------------------------------------+---------+---------+------------------+-------+----------------+
    |  0  | x linear velocity of the base (torso)         | -Inf    | Inf     | base             | free  | velocity (m/s) |
    |  1  | y linear velocity of the base (torso)         | -Inf    | Inf     | base             | free  | velocity (m/s) |
    |  2  | z linear velocity of the base (torso)         | -Inf    | Inf     | base             | free  | velocity (m/s) |
    |  3  | x angular velocity of the base (global frame) | -Inf    | Inf     | base             | free  | angular vel    |
    |  4  | y angular velocity of the base (global frame) | -Inf    | Inf     | base             | free  | angular vel    |
    |  5  | z angular velocity of the base (global frame) | -Inf    | Inf     | base             | free  | angular vel    |
    |  6  | angular velocity of FL abduction / hip roll   | -Inf    | Inf     | FL_hip_joint     | hinge | angular vel    |
    |  7  | angular velocity of FL thigh / hip pitch      | -Inf    | Inf     | FL_thigh_joint   | hinge | angular vel    |
    |  8  | angular velocity of FL calf / knee pitch      | -Inf    | Inf     | FL_calf_joint    | hinge | angular vel    |
    |  9  | angular velocity of FR abduction / hip roll   | -Inf    | Inf     | FR_hip_joint     | hinge | angular vel    |
    | 10  | angular velocity of FR thigh / hip pitch      | -Inf    | Inf     | FR_thigh_joint   | hinge | angular vel    |
    | 11  | angular velocity of FR calf / knee pitch      | -Inf    | Inf     | FR_calf_joint    | hinge | angular vel    |
    | 12  | angular velocity of RL abduction / hip roll   | -Inf    | Inf     | RL_hip_joint     | hinge | angular vel    |
    | 13  | angular velocity of RL thigh / hip pitch      | -Inf    | Inf     | RL_thigh_joint   | hinge | angular vel    |
    | 14  | angular velocity of RL calf / knee pitch      | -Inf    | Inf     | RL_calf_joint    | hinge | angular vel    |
    | 15  | angular velocity of RR abduction / hip roll   | -Inf    | Inf     | RR_hip_joint     | hinge | angular vel    |
    | 16  | angular velocity of RR thigh / hip pitch      | -Inf    | Inf     | RR_thigh_joint   | hinge | angular vel    |
    | 17  | angular velocity of RR calf / knee pitch      | -Inf    | Inf     | RR_calf_joint    | hinge | angular vel    |
    +-----+-----------------------------------------------+---------+---------+------------------+-------+----------------+
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        frame_skip: int = 10,
        reset_noise_scale: float = 0.1,
        action_rate_cost_weight: float = 0.01,
        contact_force_weight: float = 0.0,
        pose_similarity_cost_weight: float = 5e-2,
        z_error_cost_weight: float = 1.0,
        z_velocity_cost_weight: float = 0.4,
        roll_pitch_cost_weight: float = 1.0,
        wz_tracking_reward_weight: float = 1.0,
        xy_velocity_tracking_reward_weight: float = 1.5,
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        wz_error_scale: float = 0.5,
        xy_velocity_error_scale: float = 0.5,
        nominal_kp: float = 35.0,
        nominal_kd: float = 0.6,
        kp_random_scale: float = 0.15,
        kd_random_scale: float = 0.20,
        **kwargs,
    ):
        xml_file = str(
            files("legged_rl").joinpath("mujoco_menagerie/unitree_go2/scene.xml")
        )
        RecordConstructorArgs.__init__(
            self,
            frame_skip=frame_skip,
            reset_noise_scale=reset_noise_scale,
            action_rate_cost_weight=action_rate_cost_weight,
            contact_force_weight=contact_force_weight,
            pose_similarity_cost_weight=pose_similarity_cost_weight,
            z_error_cost_weight=z_error_cost_weight,
            z_velocity_cost_weight=z_velocity_cost_weight,
            roll_pitch_cost_weight=roll_pitch_cost_weight,
            wz_tracking_reward_weight=wz_tracking_reward_weight,
            xy_velocity_tracking_reward_weight=xy_velocity_tracking_reward_weight,
            contact_force_range=contact_force_range,
            wz_error_scale=wz_error_scale,
            xy_velocity_error_scale=xy_velocity_error_scale,
            nominal_kp=nominal_kp,
            nominal_kd=nominal_kd,
            kp_random_scale=kp_random_scale,
            kd_random_scale=kd_random_scale,
            **kwargs,
        )
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=None, **kwargs)

        self.metadata = {
            "render_modes": [
                "human",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self._reset_noise_scale = reset_noise_scale
        self._action_rate_cost_weight = action_rate_cost_weight
        self._contact_cost_weight = contact_force_weight
        self._pose_sim_cost_weight = pose_similarity_cost_weight
        self._z_error_cost_weight = z_error_cost_weight
        self._z_vel_cost_weight = z_velocity_cost_weight
        self._roll_pitch_cost_weight = roll_pitch_cost_weight
        self._wz_track_rew_weight = wz_tracking_reward_weight
        self._vxy_track_rew_weight = xy_velocity_tracking_reward_weight
        self._contact_force_range = contact_force_range
        self._wz_err_scale = wz_error_scale
        self._vxy_err_scale = xy_velocity_error_scale
        self._nominal_kp = nominal_kp
        self._nominal_kd = nominal_kd
        self._kp_rnd_scale = kp_random_scale
        self._kd_rnd_scale = kd_random_scale
        self._main_body = 1

        self.max_vx_cmd = 1.0
        self.max_vy_cmd = 0.5
        self.max_wz_cmd = 1.0
        self.max_vel_cmd = np.array(
            [self.max_vx_cmd, self.max_vy_cmd, self.max_wz_cmd],
            dtype=np.float32,
        )

        self.min_z = 0.1
        self.max_z = 1.0
        self.min_z_cmd = 0.2
        self.max_z_cmd = 0.5
        self.max_roll = np.deg2rad(36)
        self.max_pitch = np.deg2rad(36)
        self.max_ep_time = 10

        self._prev_action = np.zeros((12,), dtype=np.float32)
        self._vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx,vy,wz
        self._z_cmd = 0.3
        self._ep_start_time = copy(self.data.time)
        self._n_steps = 0
        self._jhome = self.init_qpos[7:]
        self._jmin = self.model.jnt_range[1:, 0]
        self._jmax = self.model.jnt_range[1:, 1]
        self._jscale = self._jmax - self._jmin
        self._kp = nominal_kp
        self._kd = nominal_kd
        self._tb_logs = {}
        self._vx_cmds = np.arange(-self.max_vx_cmd, self.max_vx_cmd + 0.5, 0.5)
        self._vy_cmds = np.arange(-self.max_vy_cmd, self.max_vy_cmd + 0.25, 0.25)
        self._wz_cmds = np.arange(-self.max_wz_cmd, self.max_wz_cmd + 0.5, 0.5)
        self._action_rate_cost_weight_max = action_rate_cost_weight
        self._pose_sim_cost_weight_max = pose_similarity_cost_weight
        self._z_vel_cost_weight_max = z_velocity_cost_weight

        self.action_space = Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation space without world coordinates and yaw
        # Plus reference velocity and torso height
        # Plus previous actions
        self.obs_size = self.data.qpos.size + self.data.qvel.size - 5 + 4 + 12
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float64
        )

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        dj_action = self._jmin + (action + 1.0) * 0.5 * self._jscale
        jtarget = np.clip(self._jhome + dj_action, self._jmin, self._jmax)

        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(jtarget, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_vel = (xy_position_after - xy_position_before) / self.dt
        x_vel, y_vel = xy_vel

        reward, reward_info = self._get_reward(action, jtarget, x_vel, y_vel)
        hs = self._get_health_state()
        obs = self._get_obs()

        wz_cmd_err = self._vel_cmd[2] - self.data.qvel[5]
        info = {
            "step/vx_cmd_error": abs(self._vel_cmd[0] - x_vel),
            "step/vy_cmd_error": abs(self._vel_cmd[1] - y_vel),
            "step/wz_cmd_error": abs(wz_cmd_err),
            "step/body_height_cmd_error": abs(self._z_cmd - self.data.qpos[2]),
            **reward_info,
        }

        self._n_steps += 1
        self._tb_logs.update(info)
        self._prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return obs, reward, hs.terminated, hs.truncated, info

    def do_simulation(self, ctrl, n_frames) -> None:
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )

        for _ in range(n_frames):
            jpos = self.data.qpos[7:]
            jvel = self.data.qvel[6:]

            jerr = ctrl - jpos
            torque = self._kp * jerr - self._kd * jvel

            self.data.ctrl[:] = torque

            mujoco.mj_step(self.model, self.data)

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()

        r, p, _ = quat2euler(qpos[3:7])

        pos: list[float] = qpos[7:].tolist()
        pos.insert(0, p)
        pos.insert(0, r)

        obs = np.concatenate(
            (pos, qvel, self._vel_cmd, (self._z_cmd,), self._prev_action)
        ).astype(np.float64)
        assert len(obs) == self.obs_size
        return obs

    def _get_health_state(self) -> HealthState:
        ep_time = self.data.time - self._ep_start_time
        if ep_time >= self.max_ep_time:
            return HealthState.TRUNCATED

        state = self.state_vector()

        z = state[2]
        r, p, _ = quat2euler(state[3:7])

        if (
            np.isfinite(state).all()
            and abs(r) < self.max_roll
            and abs(p) < self.max_pitch
            and self.min_z <= z <= self.max_z
        ):
            return HealthState.ALIVE

        return HealthState.TERMINATED

    def action_rate_cost(self, action):
        da = action - self._prev_action
        return self._action_rate_cost_weight * np.sum(np.square(da))

    def pose_similarity_cost(self, jtarget):
        dj = jtarget - self._jhome
        return self._pose_sim_cost_weight * np.sum(np.square(dj))

    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    def contact_cost(self):
        return self._contact_cost_weight * np.sum(np.square(self.contact_forces()))

    def wz_reward(self):
        wz_err_vec = self._vel_cmd[2] - self.data.qvel[5]
        wz_err = np.sum(np.square(wz_err_vec)) / self._wz_err_scale
        return self._wz_track_rew_weight * np.exp(-wz_err)

    def xy_vel_reward(self, x_vel, y_vel):
        vxy_err_vec = self._vel_cmd[:2] - np.array([x_vel, y_vel])
        vxy_err = np.sum(np.square(vxy_err_vec)) / self._vxy_err_scale
        return self._vxy_track_rew_weight * np.exp(-vxy_err)

    def _get_reward(self, action, jtarget, x_vel, y_vel):
        qpos = self.data.qpos
        qvel = self.data.qvel
        r, p, _ = quat2euler(qpos[3:7])

        ac_cost = self.action_rate_cost(action)
        ps_cost = self.pose_similarity_cost(jtarget)
        co_cost = self.contact_cost()
        ze_cost = self._z_error_cost_weight * (qpos[2] - self._z_cmd) ** 2
        zv_cost = self._z_vel_cost_weight * qvel[2] ** 2
        rp_cost = self._roll_pitch_cost_weight * (r**2 + p**2)

        wz_reward = self.wz_reward()
        xy_vel_reward = self.xy_vel_reward(x_vel, y_vel)

        costs = ac_cost + ps_cost + ze_cost + co_cost + zv_cost + rp_cost
        rewards = wz_reward + xy_vel_reward

        reward = rewards - costs
        reward_info = {
            "reward/wz_tracking": wz_reward,
            "reward/xy_velocity_tracking": xy_vel_reward,
            "reward/action_rate": -ac_cost,
            "reward/pose_similarity": -ps_cost,
            "reward/contact": -co_cost,
            "reward/z_tracking_error": -ze_cost,
            "reward/z_velocity": -zv_cost,
            "reward/roll_pitch": -rp_cost,
        }

        return reward, reward_info

    def reset_model(self):
        # Reset internal states
        self._prev_action = np.zeros((12,), dtype=np.float32)
        self._ep_start_time = copy(self.data.time)

        # Randomize initial pose
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        # Randomize PD joint regulator
        scale_kp = self.np_random.uniform(
            1 - self._kp_rnd_scale, 1 + self._kp_rnd_scale
        )
        scale_kd = self.np_random.uniform(
            1 - self._kd_rnd_scale, 1 + self._kd_rnd_scale
        )

        self._kp = self._nominal_kp * scale_kp
        self._kd = self._nominal_kd * scale_kd

        # Randomize command velocity and body height
        self._vel_cmd = np.array(
            (
                self.np_random.choice(self._vx_cmds, shuffle=False),
                self.np_random.choice(self._vy_cmds, shuffle=False),
                self.np_random.choice(self._wz_cmds, shuffle=False),
            ),
            dtype=np.float32,
        )
        self._z_cmd = self.np_random.uniform(self.min_z_cmd, self.max_z_cmd)

        return self._get_obs()

    def clear_logs(self):
        self._tb_logs.clear()

    @property
    def tb_logs(self):
        self._tb_logs["episode/time"] = self.data.time - self._ep_start_time
        return self._tb_logs.copy()

    def print_debug(self):
        lines = [
            "------------ DEBUG INFO ------------",
            f"CMD VX: {self._vel_cmd[0]:8.3f} m/s    ACTUAL VX: {self.data.qvel[0]:8.3f} m/s",
            f"CMD VY: {self._vel_cmd[1]:8.3f} m/s    ACTUAL VY: {self.data.qvel[1]:8.3f} m/s",
            f"CMD WZ: {self._vel_cmd[2]:8.3f} rad/s  ACTUAL WZ: {self.data.qvel[5]:8.3f} rad/s",
            f"CMD Z : {self._z_cmd:8.3f} m      ACTUAL Z : {self.data.qpos[2]:8.3f} m",
            "------------------------------------",
            "",
        ]
        print("\n".join(lines))

    def set_pose_similarity_cost_weight(self, weight: float):
        weight = min(weight, self._pose_sim_cost_weight_max)
        self._pose_sim_cost_weight = float(weight)

    def get_pose_similarity_cost_weight(self):
        return self._pose_sim_cost_weight

    def set_action_rate_cost_weight(self, weight: float):
        weight = min(weight, self._action_rate_cost_weight_max)
        self._action_rate_cost_weight = float(weight)

    def get_action_rate_cost_weight(self):
        return self._action_rate_cost_weight

    def set_z_vel_cost_weight(self, weight: float):
        weight = min(weight, self._z_vel_cost_weight_max)
        self._z_vel_cost_weight = float(weight)

    def get_z_vel_cost_weight(self):
        return self._z_vel_cost_weight
