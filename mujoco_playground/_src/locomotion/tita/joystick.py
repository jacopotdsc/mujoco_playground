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
"""Joystick task for Tita."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.tita import base as tita_base
from mujoco_playground._src.locomotion.tita import tita_constants as consts

import mpx.config.config_dfcip as config
from mpx.utils.mpc_wrapper_dfcip import BatchedMPCControllerWrapper
import mpx.utils.sim as sim_utils
import mpx.utils.mpc_utils as mpc_utils

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
"""Utilities for extracting collision information."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  """Get the distance and normal of the collision between two geoms."""
  mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
  idx = jnp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  """Return True if the two geoms are colliding."""
  return get_collision_info(state._impl.contact, geom1, geom2)[0] < 0  # pylint: disable=protected-access

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,#2,
      sim_dt=0.002,
      episode_length=1000,
      Kp=35.0,
      Kd=10.0,
      action_repeat=1,
      action_scale=1.0,
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
              # tracking
              tracking_lin_vel=1.0, 
              tracking_ang_vel=1.0, 
              # base
              lin_vel_z=0.0,
              ang_vel_xy=0.0,
              orientation=1.0,
              base_height=0.5,
              # Other.
              dof_pos_limits=-0.1,
              pose=0.0,
              # Other.
              termination=-1.0,
              stand_still=-0.0,
              # Regularization.
              torques=-0.00005,
              action_rate=-0.001, 
              energy=-0.0001,
              # Feet.
              #feet_clearance=-2.0,
              #feet_height=-0.2,
              #feet_slip=-0.1,
              #feet_air_time=0.1,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.1,
          base_height_target=0.40,
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[2.0, 0.0, 0.5],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      impl="jax",
      naconmax=4 * 8192,
      njmax=40,
  )


class Joystick(tita_base.TitaEnv):
  """Track a joystick command for the Tita Wheel-Legged Robot."""

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

    # Limit DOFs, exclude body and wheels. First joint is freejoint.
    jnt_range_reduced = jp.delete(self.mj_model.jnt_range, jp.array([0, 4, 8]), axis=0)

    self._lowers, self._uppers = jnt_range_reduced.T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # Setup Indices and IDs
    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = jp.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = jp.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
    self._base_com_adr = self._sensor_adr("base_subtree_com")
    self._base_com_linvel_adr = self._sensor_adr("base_subtree_linvel")

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    sim_frequency = 1.0 / self.sim_dt
    self.mpc_period = max(1, int(sim_frequency / config.mpc_frequency))
    self.mpc = BatchedMPCControllerWrapper(config, 1)
    self._wheel_radius = jp.array([self._mj_model.geom(name).size[0] for name in consts.FEET_GEOMS])

  def build_tita_state(self, data: mjx.Data) -> jax.Array:
    # CoM pos/vel dai sensori subtree (già JAX)
    pcom = data.sensordata[self._base_com_adr]          # (3,)
    vcom = data.sensordata[self._base_com_linvel_adr]   # (3,)

    # Centri e rotazioni dei geom ruota nel world, da mjx.Data
    centers = data.geom_xpos[self._feet_geom_id]              # (2, 3)
    Rs = data.geom_xmat[self._feet_geom_id].reshape(2, 3, 3)  # (2, 3, 3)

    # Offset del contact point (rCP) per ogni ruota.
    # get_rCP deve essere jnp-based (niente numpy/float() interni).
    l_rcp = mpc_utils.get_rCP(Rs[0], self._wheel_radius[0])
    r_rcp = mpc_utils.get_rCP(Rs[1], self._wheel_radius[1])

    pl_world = centers[0] + l_rcp
    pr_world = centers[1] + r_rcp

    # Velocità piedi nel world dai sensori framelinvel
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]  # (2, 3)
    dpl_world, dpr_world = feet_vel[0], feet_vel[1]

    return jp.concatenate([pcom, vcom, pl_world, pr_world, dpl_world, dpr_world])

  def get_dfip_current_state(self, tita_state: jax.Array, theta_prev):
    def unwrap_near(theta_wrapped, theta_prev):
        a = theta_wrapped - theta_prev
        a = (a + jp.pi) % (2 * jp.pi)
        a = jp.where(a < 0, a + 2 * jp.pi, a)
        a = a - jp.pi
        return theta_prev + a

    pcom      = tita_state[0:3]
    vcom      = tita_state[3:6]
    pl_world  = tita_state[6:9]
    pr_world  = tita_state[9:12]
    dpl_world = tita_state[12:15]
    dpr_world = tita_state[15:18]

    c_world  = (pl_world + pr_world) / 2.0
    vc_world = (dpl_world + dpr_world) / 2.0

    diff = pl_world - pr_world
    theta_wrapped = jp.arctan2(-diff[0], diff[1])
    theta = unwrap_near(theta_wrapped, theta_prev)

    R = jp.array([
        [jp.cos(theta), -jp.sin(theta), 0.0],
        [jp.sin(theta),  jp.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])

    dpl_body = R.T @ dpl_world
    dpr_body = R.T @ dpr_world

    d = config.d
    w = (dpr_body[0] - dpl_body[0]) / d
    v = (dpr_body[0] + dpl_body[0]) / 2.0

    x0 = jp.concatenate([
        pcom,
        vcom,
        c_world,
        jp.array([vc_world[2]]),
        jp.array([theta]),
        jp.array([v]),
        jp.array([w]),
    ])
    return x0, theta
   

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Base randomization
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    #qpos = qpos.at[0:2].set(qpos[0:2] + dxy) 
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-jp.pi, maxval=jp.pi)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    #qpos = qpos.at[3:7].set(new_quat) 

    # velocity randomization: d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    #qvel = qvel.at[0:6].set(
    #    jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    #)

    data = data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=jp.zeros(self.mjx_model.nu),
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


    mpc_state = self.mpc.init_state()
    mpc_state, tita_state, dfcip_state, tau, qddot, fl, fr, desired, theta_prev = self._run_mpc_wbc(
        data=data, 
        qpos=data.qpos, 
        qvel=data.qvel, 
        command=cmd,
        mpc_state=mpc_state,
        theta_prev=0.0,
        timestep=0
        )


    info = {
        "rng": rng,
        "command": cmd,
        "target_command": cmd,
        "theta_prev": theta_prev,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        #"feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        #"swing_peak": jp.zeros(2),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        "step_counter": 0,
        "mpc_state": mpc_state,
        "mpc_tau": tau,
        "dfcip_state": dfcip_state,
        "low_level_controller": {
            #"tau_ff":      jp.zeros(self.mjx_model.nu),
            "q_des":       jp.zeros(self.mjx_model.nu),
            "dq_des":      jp.zeros(self.mjx_model.nu),
            "qacc_joints": jp.zeros(self.mjx_model.nu),
            "tau_p":       jp.zeros(self.mjx_model.nu),
            "tau_d":       jp.zeros(self.mjx_model.nu),
            "kp":          jp.zeros(()),
            "kd":          jp.zeros(()),
            "action_scale": jp.zeros(()),
            "action":      jp.zeros(self.mjx_model.nu),
            "qpos":        jp.zeros(self.mjx_model.nq),
        },
        "use_only_mpc": False,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    #metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)
  
  def _run_mpc_wbc(self, data, qpos: jax.Array, qvel: jax.Array, command: jax.Array, mpc_state, theta_prev, timestep: int):
    """Run the MPC planner (called every mpc_period steps)."""
    qpos = jp.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
    qvel = jp.nan_to_num(qvel, nan=0.0, posinf=0.0, neginf=0.0)

    contact_ids = sim_utils.geom_ids(self._mj_model, config.contact_frame)
    
    tita_state = self.build_tita_state(data)
    x0, theta_prev = self.get_dfip_current_state(tita_state, theta_prev)

    mpc_command = jp.array([command[0], command[1], command[2], 0.4])
    mpc_state, reference = self.mpc.run(mpc_state, x0[None, :], mpc_command[None, :])

    pl_world_wbc = tita_state[6:9][None, :]
    pr_world_wbc = tita_state[9:12][None, :]
    dpl_world_wbc = tita_state[12:15][None, :]
    dpr_world_wbc = tita_state[15:18][None, :]

    mpc_state, tau, qddot, fl, fr, desired = self.mpc.whole_body_run(
        mpc_state,
        x0,
        jnp.asarray(qpos)[None, :],
        jnp.asarray(qvel)[None, :],
        pl_world_wbc,
        pr_world_wbc,
        dpl_world_wbc,
        dpr_world_wbc,
    )

    tau = jp.nan_to_num(tau[0], nan=0.0, posinf=0.0, neginf=0.0)
    return mpc_state, tita_state, x0, tau, qddot, fl, fr, desired, theta_prev

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)

    mpc_state, tita_state, dfcip_state, tau, qddot, fl, fr, desired, theta_prev = self._run_mpc_wbc(
        data=state.data, 
        qpos=state.data.qpos, 
        qvel=state.data.qvel, 
        command=state.info["command"],
        mpc_state=state.info["mpc_state"],
        theta_prev=state.info["theta_prev"],
        timestep=state.info["step_counter"]
        )
    state.info["mpc_state"] = mpc_state
    state.info["mpc_tau"] = tau
    state.info["theta_prev"] = theta_prev
    state.info["dfcip_state"] = dfcip_state
    
    def substep_fn(data, _):
        tau_p = self._config.Kp * ( self._default_pose + action*self._config.action_scale - data.qpos[7:])
        tau_d = self._config.Kd * ( - data.qvel[6:])
        tau_network = tau_p.at[jnp.array([3, 7])].set(0.0) + tau_d
        tau_network = tau_network*(1 - state.info["use_only_mpc"].astype(tau.dtype))

        motor_targets = tau + tau_network

        data = data.replace(ctrl=motor_targets)
        data = mjx.step(self.mjx_model, data)
        llc_log = {
            #"tau_ff":      tau_ff,
            "q_des":       jp.zeros_like(data.qpos[7:]),  # q_des,
            "dq_des":      jp.zeros_like(data.qvel[6:]),  # dq_des,
            "qacc_joints": jp.zeros_like(data.qvel[6:]),  # qacc_joints,
            "tau_p":       tau_p,
            "tau_d":       tau_d,
            "kp":          jp.array(self._config.Kp),
            "kd":          jp.array(self._config.Kd),
            "action_scale": jp.array(self._config.action_scale),
            "action":      action,
            "qpos":        data.qpos,
        }
        return data, llc_log
    
    data, llc_logs = jax.lax.scan(substep_fn, state.data, None, length=self.n_substeps)
    state.info["low_level_controller"] = jax.tree_util.tree_map(lambda x: x[-1], llc_logs)
    state = state.replace(data=data)

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    #contact_filt = contact | state.info["last_contact"]
    #first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    #state.info["feet_air_time"] += self.dt
    #p_f = data.site_xpos[self._feet_site_id]
    #p_fz = p_f[..., -1]
    #state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["target_command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["target_command"]),
        state.info["target_command"],
    )
    # Exponential smoothing: tau ~ 0.4s at ctrl_dt=0.02s
    state.info["command"] = (
        state.info["command"]
        + 0.005 * (state.info["target_command"] - state.info["command"])
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    #state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    #state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    #state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    state.info["step_counter"] += 1

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    floating_base_touch_ground = geoms_colliding(data, self._torso_body_id, self._floor_geom_id)
    nan_qpos = jp.any(jp.isnan(data.qpos))
    nan_qvel = jp.any(jp.isnan(data.qvel))
    return (
        fall_termination | floating_base_touch_ground | nan_qpos | nan_qvel
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
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

    dfcip_state = info["dfcip_state"]
    noisy_dfcip_state = (
        dfcip_state
        + (2 * jax.random.uniform(noise_rng, shape=dfcip_state.shape) - 1)
        * self._config.noise_config.level
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        info["command"],  # 3
        noisy_dfcip_state,  # 18
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
        #info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
        dfcip_state,  # 18
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
      #first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    current_up = data.site_xmat[self._imu_site_id] @ jp.array([0.0, 0.0, 1.0])
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        #"orientation": self._cost_orientation(self.get_upvector(data)),
        "base_height": self._reward_height(data.qpos[2]),
        "orientation": self._reward_orientation(current_up, jp.array([0, 0, 1])),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        #"feet_slip": self._cost_feet_slip(data, contact, info),
        #"feet_clearance": self._cost_feet_clearance(data),
        #"feet_height": self._cost_feet_height(
        #    info["swing_peak"], first_contact, info
        #),
        #"feet_air_time": self._reward_feet_air_time(
        #    info["feet_air_time"], first_contact, info["command"]
        #),
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

  # Base related rewards
  def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _reward_orientation(
      self, current_up_vec: jax.Array, target_up_vec: jax.Array
  ) -> jax.Array:
    cos_dist = jp.dot(current_up_vec, target_up_vec)
    normalized = 0.5 * cos_dist + 0.5
    return jp.square(normalized)
  
  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _reward_height(self, body_height: jax.Array) -> jax.Array:
    error = self._init_q[2] - body_height 
    return jp.exp(-error / 1.0)

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
    weight = jp.array([1.0, 1.0, 1.0, 0.0] * 2)
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
    qpos_reduced = jp.delete(qpos, consts.TITA_WHEEL_INDICES)  # exclude wheels
    out_of_limits = -jp.clip(qpos_reduced - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos_reduced - self._soft_uppers, 0.0, None)
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
