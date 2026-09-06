# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Lite3."""

from typing import Any, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
import mujoco

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.lite3 import base as lite3_base
from mujoco_playground._src.locomotion.lite3 import lite3_constants as consts

#import mpx.config.config_srbd as config
#from mpx.utils.mpc_wrapper_srbd import BatchedMPCControllerWrapper

def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  """Get the distance and normal of the collision between two geoms."""
  mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
  idx = jp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  """Return True if the two geoms are colliding."""
  return get_collision_info(state._impl.contact, geom1, geom2)[0] < 0  # pylint: disable=protected-access


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.002,
      episode_length=1000,
      # Lite3 weighs 11.94 kg, the same class as Go1 (12.0 kg), which Playground
      # runs at Kp=35/Kd=0.5. Aliengo's 70/1.0 is sized for 22 kg and would put
      # Kp*action_scale = 35 N.m past the actuators' +-30 N.m ctrlrange.
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base reward.
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              # Other.
              dof_pos_limits=-1.0,
              pose=0.5,
              # Other.
              termination=-1.0,
              stand_still=-1.0,
              # Regularization.
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              # Feet.
              feet_clearance=-2.0,
              feet_height=-0.2,
              feet_slip=-0.1,
              feet_air_time=0.1,
              # Penalises a foot parked in the air. Without it a foot that
              # never touches down costs nothing: feet_air_time and feet_height
              # are only paid at first_contact, and feet_clearance is minimised
              # by hovering at exactly max_foot_height.
              feet_air_time_limit=-1.0,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.1,
          # A normal swing lasts 0.05-0.13 s; 0.5 s is ~4-10x that, so this
          # only fires on a foot that has stopped stepping altogether.
          max_air_time=0.5,
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[1.5, 0.8, 1.2],
          #a=[0.0, 0.0, 0.0],  # Set to 0.0 to disable command.
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      impl="jax",
      naconmax=4 * 8192,
      njmax=40,
  )


class Joystick(lite3_base.Lite3Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if task.startswith("rough") or task in ("stairs_terrain", "perlin_terrain"):
      config.naconmax = 8 * 8192
      config.njmax = 12 + 48
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    # Lite3's knee range is strictly positive ([0.524, 2.792]), so scaling the
    # bounds by the factor (the Aliengo/Go1 idiom) would push the soft lower
    # bound *outside* the hard limit. Shrink around the range center instead.
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    self._termination_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.TERMINATION_GEOMS]
    )
    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    #sim_frequency = 1.0 / self.sim_dt
    #self.mpc_period = max(1, int(sim_frequency / config.mpc_frequency))
    #self.mpc = BatchedMPCControllerWrapper(config, 1)

    # --- print initial robot state ---
    _d = mujoco.MjData(self._mj_model)
    _d.qpos[:] = np.array(self._init_q)
    mujoco.mj_forward(self._mj_model, _d)
    _sep = "=" * 55
    print(_sep)
    print("Lite3 – initial robot state")
    print(_sep)
    print(f"  p0    (xyz)    : {_d.qpos[0:3]}")
    print(f"  quat0 (w,x,y,z): {_d.qpos[3:7]}")
    print(f"  q0    (joints) : {_d.qpos[7:]}")
    print(f"  Total mass     : {self._torso_mass:.4f} kg")
    # full 3x3 inertia in body frame: I = R @ diag(principal) @ R.T
    # body_iquat is (w,x,y,z) – rotation from principal frame to body frame
    _iq = self._mj_model.body_iquat[self._torso_body_id]  # (w,x,y,z)
    _w, _x, _y, _z = _iq
    _R = np.array([
        [1-2*(_y*_y+_z*_z),  2*(_x*_y-_z*_w),  2*(_x*_z+_y*_w)],
        [  2*(_x*_y+_z*_w),1-2*(_x*_x+_z*_z),  2*(_y*_z-_x*_w)],
        [  2*(_x*_z-_y*_w),  2*(_y*_z+_x*_w),1-2*(_x*_x+_y*_y)],
    ])
    _Idiag = np.diag(self._mj_model.body_inertia[self._torso_body_id])
    _I3 = _R @ _Idiag @ _R.T
    print("  Torso inertia (3x3 body frame) [kg·m²]:")
    for _row in _I3:
      print(f"    [{_row[0]:+.6f}  {_row[1]:+.6f}  {_row[2]:+.6f}]")
    _com = _d.subtree_com[self._torso_body_id]
    print(f"  COM pos (world): [{_com[0]:+.4f}, {_com[1]:+.4f}, {_com[2]:+.4f}]")
    print(f"  COM height     : {_com[2]:.4f} m")

    M = np.zeros((self._mj_model.nv, self._mj_model.nv))
    mujoco.mj_fullM(self._mj_model, M, _d.qM)

    print(M.shape)
    print(M[3:6, 3:6])

    print(f"  Floating base  : free joint → qpos[0:3]=xyz, qpos[3:7]=quat(wxyz)")
    print("  Foot positions (world frame):")
    for _name, _idx in zip(consts.FEET_SITES, self._feet_site_id):
      _p = _d.site_xpos[_idx]
      print(f"    {_name:<25s}: [{_p[0]:+.4f}, {_p[1]:+.4f}, {_p[2]:+.4f}]")
    print(_sep)
  
  '''
  def _build_mpc_input(self, command: jax.Array) -> jax.Array:
    command = jp.nan_to_num(command, nan=0.0, posinf=0.0, neginf=0.0)
    return jp.array([
        command[0],
        command[1],
        0.0,
        0.0,
        0.0,
        command[2],
        config.robot_height,
    ])
  
  def _run_mpc(self, qpos: jax.Array, qvel: jax.Array, geom_xpos: jax.Array,
               command: jax.Array, mpc_state):
    """Run the MPC planner (called every mpc_period steps)."""
    qpos = jp.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
    qvel = jp.nan_to_num(qvel, nan=0.0, posinf=0.0, neginf=0.0)
 
    foot_world = jp.stack([geom_xpos[gid] for gid in self._feet_geom_id], axis=0)
    foot_pos = foot_world.reshape(1, -1)
    foot_pos = jp.nan_to_num(foot_pos, nan=0.0, posinf=0.0, neginf=0.0)
 
    x0 = jp.concatenate([qpos[:3], qpos[3:7], qvel[:3], qvel[3:6]])[None]
    mpc_input = self._build_mpc_input(command)[None, :]
 
    contact = (foot_world[:, 2] < 0.035).astype(jp.float32)[None, :]
 
    mpc_state = self.mpc.run(mpc_state, x0, mpc_input, foot_pos, contact)
    return mpc_state


  def _whole_body_ctrl(self, qpos: jax.Array, qvel: jax.Array, mpc_state):
    """Run whole-body controller (called every sim step)."""
    qpos = jp.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
    qvel = jp.nan_to_num(qvel, nan=0.0, posinf=0.0, neginf=0.0)
    tau, _ = self.mpc.whole_body_run(mpc_state, qpos[None], qvel[None])
    tau = jp.nan_to_num(tau[0], nan=0.0, posinf=0.0, neginf=0.0)
    return tau
  '''

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    cmd = jax.random.uniform(
        key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )

    #mpc_state  = self.mpc.init_state()
    #mpc_state = self._run_mpc(
    #    data.qpos, data.qvel, data.geom_xpos, cmd, mpc_state
    #)
    #tau = self._whole_body_ctrl(data.qpos, data.qvel, mpc_state)


    info = {
        "rng": rng,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "target_command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        #"mpc_state": mpc_state,
        #"mpc_tau": tau,
        "reward_terms" : {}
    }

    dummy_rewards = self._get_reward(
        data,
        jp.zeros(self.mjx_model.nu),
        info,
        {},
        jp.array(False),
        jp.array(False),
        jp.zeros(len(self._feet_geom_id), dtype=bool),
    )

    dummy_rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in dummy_rewards.items()
    }

    info["reward_terms"] = dummy_rewards


    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
  #   qpos = state.data.qpos
  #   new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
  #   new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
  #   qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
  #   state = state.replace(data=state.data.replace(qpos=qpos))
  #   return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)

    #motor_targets = self._default_pose + action * self._config.action_scale
    #target_pos = self._default_pose + action * self._config.action_scale
    #motor_targets = 60*(target_pos - state.data.qpos[7:]) + 1.0 * ( -state.data.qvel[6:])
    #data = mjx_env.step(
    #    self.mjx_model, state.data, motor_targets, self.n_substeps
    #)

    #mpc_state = self._run_mpc(
    #    state.data.qpos, state.data.qvel, state.data.geom_xpos,
    #    state.info["command"], state.info["mpc_state"],
    #)
    #state.info["mpc_state"] = mpc_state

    def substep_fn(data, _):
        #tau_ff = self._whole_body_ctrl(data.qpos, data.qvel, mpc_state)

        target_pos = self._default_pose + action * self._config.action_scale
        tau_pd = self._config.Kp * (target_pos - data.qpos[7:]) + self._config.Kd * (-data.qvel[6:])
        
        motor_targets = tau_pd #tau_ff + tau_pd

        data = data.replace(ctrl=motor_targets)
        data = mjx.step(self.mjx_model, data)
        return data, None

    data, _ = jax.lax.scan(substep_fn, state.data, None, length=self.n_substeps)
    state = state.replace(data=data)

    #motor_targets = self._default_pose + action * self._config.action_scale
    #data = mjx_env.step(
    #    self.mjx_model, state.data, motor_targets, self.n_substeps
    #)
    
    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["reward_terms"] = rewards
    
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["target_command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["target_command"]),
        state.info["target_command"],
    )
    # Exponential smoothing toward the setpoint, same as lite3/joystick.py.
    # Keeping "target_command" as the setpoint is also what lets an eval
    # harness pin a fixed command (train_srbd.py --cmd re-injects it each step).
    state.info["command"] = (
        state.info["command"]
        + 0.05 * (state.info["target_command"] - state.info["command"])
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    base_contact = jp.array([
        geoms_colliding(data, gid, self._floor_geom_id)
        for gid in self._termination_geom_id
    ]).any()
    return fall_termination | base_contact

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      action: Optional[jax.Array] = None,
  ) -> Dict[str, jax.Array]:
    # `action` is the previous action. It defaults to info["last_act"] so the
    # env's own reset/step calls stay unchanged; eval harnesses that recompute
    # the obs after patching info (train_srbd.py's --cmd path) pass it
    # explicitly, matching the Tita envs' signature.
    last_act = info["last_act"] if action is None else action
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        last_act,  # 12
        info["command"],  # 3
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        joint_angles - self._default_pose,  # 12
        joint_vel,  # 12
        data.actuator_force,  # 12
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_air_time_limit": self._cost_feet_air_time_limit(
            info["feet_air_time"], info["command"]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }

  # Tracking rewards.

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  # Other rewards.

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose.
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    return rew_air_time

  def _cost_feet_air_time_limit(
      self, air_time: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Penalise air time beyond a normal swing, per foot and at every step.
    # Unlike feet_air_time / feet_height, this does not wait for first_contact,
    # so a foot that never touches down is charged continuously.
    cmd_norm = jp.linalg.norm(commands)
    excess = jp.clip(
        air_time - self._config.reward_config.max_air_time, 0.0, None
    )
    return jp.sum(excess) * (cmd_norm > 0.01)

  # Perturbation and command sampling.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1