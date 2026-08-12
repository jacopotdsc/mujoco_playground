# Joystick task for TITA (two-wheeled balancing biped) on MuJoCo Playground / MJX.
#
# The robot has two 3-DOF legs, each ending in a driven wheel:
#   qpos[7:] index -> 0,1,2 = left  hip/thigh/knee, 3 = left  wheel
#                     4,5,6 = right hip/thigh/knee, 7 = right wheel
#   LEG_DOF_IDS = [0,1,2,4,5,6]   WHEEL_DOF_IDS = [3,7]
#
# Control (MuJoCo actuators, gains overridden in tita/base.py; applied every sim
# substep at 500 Hz, ctrl held constant across the 5 substeps of one policy step):
#   legs   (<position> actuator): tau = 35*(q_target - q) - 10*qdot   (Kp=35, Kd=10)
#   wheels (<velocity> actuator): tau = 0.5*(v_target - qdot)         (Kd_wheel=0.5)
#   torque saturation +-120 N*m (forcerange in tita.xml)
#
# NOTE on the wheel law: with v_target = action_scale_vel*a the <velocity> actuator
# executes  tau = 0.5*(action_scale_vel*a - qdot) = (0.5*action_scale_vel)*a - 0.5*qdot,
# i.e. a DAMPED TORQUE COMMAND (feedforward torque proportional to action + velocity
# damping) - NOT a stiff velocity lock. This is the same structure as the DDT Isaac
# Gym reference wheel law  tau = 2.5*a - 0.5*qdot. Matching it exactly needs
# action_scale_vel = 5.0  (-> 2.5*a - 0.5*qdot).
#
# Action -> command:
#   legs:   q_target = default_pose + action * action_scale_pos    (+-0.5 rad)
#   wheels: v_target = action        * action_scale_vel            (5 rad/s -> feedforward 2.5*a)
#
# Timing: sim_dt = 0.002, ctrl_dt = 0.01  ->  5 substeps, 100 Hz policy.
"""Joystick task for TITA."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.tita import base as tita_base
from mujoco_playground._src.locomotion.tita import tita_constants as consts

def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  """Distance and normal of the (geom1, geom2) contact, if any."""
  mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
  idx = jp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  """True if the two geoms are colliding."""
  return get_collision_info(state._impl.contact, geom1, geom2)[0] < 0  # pylint: disable=protected-access


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.002,
      episode_length=1000,
      Kp=50.0,
      Kd=1.0,
      Kd_wheel=0.5,      # kv delle ruote (velocity control)
      action_repeat=1,
      action_scale_pos=0.5,
      # 5.0 -> executed wheel law tau = 0.5*(5*a - w) = 2.5*a - 0.5*w, IDENTICAL to the
      # DDT Isaac Gym reference (2.5*a - 0.5*w). Was 30.0, which gave 15*a - 0.5*w:
      # 6x the reference feedforward, i.e. a unit action produced ~3-4x the corrective
      # torque a moderate lean needs (~4 N*m/wheel at 0.1 rad) -> over-twitchy wheels.
      action_scale_vel=25.0,
      soft_joint_pos_limit_factor=0.95, 
      noise_config=config_dict.create(
          level=0.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=2.0,
              tracking_ang_vel=0.5,
              orientation=-2.0,
              ang_vel_xy=-0.3,
              base_height=-1.0,
              posture=-5.0,      # command-gated; see _cost_posture
              torques=-1e-4,
              action_rate=-0.01,
              dof_pos_limits=-1.0,
              termination=-5.0,
          ),
          only_positive_rewards=False,
          tracking_sigma=0.25,
          base_height_target=0.4,   # task target; home-pose COM is ~0.3956 (inside the 2 cm deadzone)
          posture_cmd_sigma=0.25,     # gate width: posture relaxes as |command| grows
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      # Command = [forward_vel (m/s), yaw_rate (rad/s)].
      command_config=config_dict.create(
          a=[1.0, 0.5],     # amplitude (uniform half-range) per command
          b=[0.75, 0.75],    # prob a resampled command stays non-zero
          p_stand=0.2,      # prob of an explicit zero (standing) command
      ),
      impl="jax",
      naconmax=4 * 8192,
      njmax=40,
  )


class Joystick(tita_base.TitaEnv):
  """Track a joystick command with the TITA wheeled biped."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    # The "home" keyframe spawns the base at z=0.44, which puts each wheel bottom at
    # -0.0035 m (3.5 mm below the floor) -> a contact impulse on every reset. Lift the
    # base by 3.5 mm so the wheels just touch. (Pose/COM unchanged: COM ~0.3956 m, still
    # inside the 2 cm height-reward deadzone around the 0.4 target.)
    self._init_q = self._init_q.at[2].set(0.4435)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._leg_ids = jp.array(consts.LEG_DOF_IDS)
    self._wheel_ids = jp.array(consts.WHEEL_DOF_IDS)

    # Soft joint limits (legs only; wheels are continuous).
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    f = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = self._lowers[self._leg_ids] * f
    self._soft_uppers = self._uppers[self._leg_ids] * f
    self._torque_limits = jp.array(self.mj_model.actuator_forcerange[:, 1])

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = jp.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom(consts.FLOOR_GEOM).id
    self._feet_geom_id = jp.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    self._termination_geom_id = jp.array(
        [self._mj_model.geom(name).id for name in consts.TERMINATION_GEOMS]
    )
    self._collision_geom_id = jp.array(
        [self._mj_model.geom(name).id for name in consts.COLLISION_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      adr = self._mj_model.sensor_adr[sensor_id]
      dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(list(range(adr, adr + dim)))
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._base_com_adr = self._sensor_adr("base_subtree_com")

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    # Posture weights over the 6 leg joints [hip, thigh, knee] x 2.
    # The hip (leg_1) is the "spread" DOF (d(track)/d(hip) ~ 0.35 m/rad), so it
    # gets the highest weight.
    self._posture_weights = jp.array([1.0, 0.5, 0.5, 1.0, 0.5, 0.5])

  # --------------------------------------------------------------------
  # Reset / step.
  # --------------------------------------------------------------------
  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x,y = +U(-0.5, 0.5), yaw = U(-pi, pi).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    #qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
    #qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

    # Start under mild planar motion. Ranges kept small for the STATIONARY task:
    # with the wheels reset to v_des=0, even vx=0.2 m/s pitches the base ~64 deg
    # in 0.5 s uncontrolled, so a large initial velocity makes every episode start
    # as a hard recovery instead of standing. Lateral (vy) self-damps via wheel
    # friction, so it is kept smaller still.
    rng, key = jax.random.split(rng)
    vx = jax.random.uniform(key, (1,), minval=-0.2, maxval=0.2)
    rng, key = jax.random.split(rng)
    vy = jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1)
    #qvel = qvel.at[0:2].set(jp.concatenate([vx, vy]))

    ctrl = jp.zeros(self.mjx_model.nu)
    ctrl = ctrl.at[self._leg_ids].set(qpos[7:][self._leg_ids])

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
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
        key2, shape=(self._cmd_a.shape[0],), minval=-self._cmd_a, maxval=self._cmd_a
    )

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "last_dof_vel": jp.zeros(consts.NUM_DOFS),
        "feet_air_time": jp.zeros(2),
        "last_feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),            # current_max_feet_height
        "last_max_feet_height": jp.zeros(2),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        "robot": {
            "qpos": data.qpos,
            "qvel": data.qvel,
            "com_height": data.sensordata[self._base_com_adr][2],
            "base_height_target": self._config.reward_config.base_height_target,
            "local_linvel": self.get_local_linvel(data),
            "gyro": self.get_gyro(data),
            "global_linvel": self.get_global_linvel(data),
            "global_angvel": self.get_global_angvel(data),
            "gravity": self.get_gravity(data),
            "upvector": self.get_upvector(data),
            "accelerometer": self.get_accelerometer(data),
            "joint_pos": data.qpos[7:],
            "joint_vel": data.qvel[6:],
            "actuator_force": data.actuator_force,
            "ctrl": data.ctrl,
            "feet_pos": data.site_xpos[self._feet_site_id],
            "feet_vel": data.sensordata[self._foot_linvel_sensor_adr],
            "ext_force": data.xfrc_applied[self._torso_body_id, :3],
            "joint_pos_3d": data.xanchor,
        },
        "reward_terms" : {}
    }

    dummy_rewards = self._get_reward(
        data,
        jp.zeros(self.mjx_model.nu),
        info,
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

    obs = self._get_obs(data, info, jp.zeros(self.mjx_model.nu))
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)

    ctrl = jp.zeros(self.mjx_model.nu)
    ctrl = ctrl.at[self._leg_ids].set(
        self._default_pose[self._leg_ids]
        + action[self._leg_ids] * self._config.action_scale_pos
    )
    ctrl = ctrl.at[self._wheel_ids].set(
        action[self._wheel_ids] * self._config.action_scale_vel
    )

    data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)
    state = state.replace(data=data)

    # Foot contact bookkeeping (kept for eval/plotting).
    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt

    foot_z = data.site_xpos[self._feet_site_id][..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], foot_z)
    state.info["last_feet_air_time"] = jp.where(
        first_contact,
        state.info["feet_air_time"],
        state.info["last_feet_air_time"],
    )
    state.info["last_max_feet_height"] = jp.where(
        first_contact,
        state.info["swing_peak"],
        state.info["last_max_feet_height"],
    )

    state.info["robot"] = {
        "qpos": data.qpos,
        "qvel": data.qvel,
        "com_height": data.sensordata[self._base_com_adr][2],
        "base_height_target": self._config.reward_config.base_height_target,
        "local_linvel": self.get_local_linvel(data),
        "gyro": self.get_gyro(data),
        "global_linvel": self.get_global_linvel(data),
        "global_angvel": self.get_global_angvel(data),
        "gravity": self.get_gravity(data),
        "upvector": self.get_upvector(data),
        "accelerometer": self.get_accelerometer(data),
        "joint_pos": data.qpos[7:],
        "joint_vel": data.qvel[6:],
        "actuator_force": data.actuator_force,
        "ctrl": data.ctrl,
        "feet_pos": data.site_xpos[self._feet_site_id],
        "feet_vel": data.sensordata[self._foot_linvel_sensor_adr],
        "ext_force": data.xfrc_applied[self._torso_body_id, :3],
        "joint_pos_3d": data.xanchor,
    }

    obs = self._get_obs(data, state.info, action)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    if self._config.reward_config.only_positive_rewards:
      reward = sum(
          v for k, v in rewards.items() if k != "termination"
      ) * self.dt
      reward = jp.clip(reward, 0.0, 10000.0)
      reward += rewards["termination"] * self.dt
    else:
      reward = jp.clip(sum(rewards.values()) * self.dt, -10000.0, 10000.0)

    state.info["reward_terms"] = rewards


    # Aggiorna i buffer (equivalente della coda di post_physics_step).
    state.info["step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_dof_vel"] = data.qvel[6:]
    state.info["feet_air_time"] *= ~contact_filt
    state.info["swing_peak"] *= ~contact_filt
    state.info["last_contact"] = contact

    # Ricampionamento comandi a intervallo fisso (Isaac: resampling_time).
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["command"]),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Isaac: contatto sui body di terminazione (base). In più: caduta.
    fall_termination = self.get_upvector(data)[-1] < 0.0
    base_contact = jp.array([
        geoms_colliding(data, gid, self._floor_geom_id)
        for gid in self._termination_geom_id
    ]).any()
    return fall_termination | base_contact

  # --------------------------------------------------------------------
  # Observations (proprio "state" + "privileged_state").
  # --------------------------------------------------------------------

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], action: jax.Array
  ) -> Dict[str, jax.Array]:
    noise = self._config.noise_config

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

    # Stessa composizione di Isaac: niente lin_vel, niente pos delle ruote.
    leg_pos_err = (noisy_joint_angles - self._default_pose)[self._leg_ids]
    # CoM-height error (com_z - target). The actor was previously BLIND to height:
    # height only lived in privileged_state (critic). Without this signal the policy
    # cannot close a loop on CoM_z and slowly sinks. Centred near 0 so the running
    # obs-normalizer behaves. Same CoM sensor used by the height reward.
    current_com_height = data.sensordata[self._base_com_adr][2]
    com_height_err = (current_com_height - self._config.reward_config.base_height_target)
    state = jp.hstack([
        noisy_linvel,        # 3   local linear velocity
        noisy_gyro,          # 3   body angular velocity
        noisy_gravity,       # 3   projected gravity (tilt)
        leg_pos_err,         # 6   leg position error vs default
        noisy_joint_vel,     # 8   all joint velocities (incl. wheels)
        action,              # 8   previous action
        info["command"],     # 2   [forward_vel, yaw_rate]
        current_com_height, #com_height_err[None], # 1  CoM height error (com_z - target)
    ])  # total: 34

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    privileged_state = jp.hstack([
        state,                                            # 33
        gyro,                                             # 3
        accelerometer,                                    # 3
        gravity,                                          # 3
        linvel,                                           # 3
        angvel,                                           # 3
        (joint_angles - self._default_pose)[self._leg_ids],  # 6
        joint_vel,                                        # 8
        data.actuator_force,                              # 8
        info["last_contact"],                             # 2
        feet_vel,                                         # 6
        info["feet_air_time"],                            # 2
        current_com_height,                        # 1  base height
        data.xfrc_applied[self._torso_body_id, :3],       # 3  external push
    ])  # total: 84

    return {"state": state, "privileged_state": privileged_state}

  # --------------------------------------------------------------------
  # Rewards.
  # --------------------------------------------------------------------
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del first_contact, contact
    command = info["command"]
    body_height = data.sensordata[self._base_com_adr][2]
    return {
        # Task tracking (raw error -> Gaussian kernel).
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            command, self.get_local_linvel(data)),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            command, self.get_gyro(data)),
        # Balance / body configuration.
        "orientation": self._cost_orientation(data),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "base_height": self._cost_height(body_height),
        "posture": self._cost_posture(data.qpos[7:], command),
        # Effort / smoothness / safety.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "termination": self._cost_termination(done),
    }

  def _reward_tracking_lin_vel(
      self, command: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    err = jp.square(command[0] - local_vel[0])
    return jp.exp(-err / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, command: jax.Array, gyro: jax.Array
  ) -> jax.Array:
    err = jp.square(command[1] - gyro[2])
    return jp.exp(-err / self._config.reward_config.tracking_sigma)

  def _cost_orientation(self, data: mjx.Data) -> jax.Array:
    # 0 when the base is level; grows with tilt.
    return jp.sum(jp.square(self.get_gravity(data)[:2]))

  def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_height(self, body_height: jax.Array) -> jax.Array:
    # Smooth, bounded cost with a NON-ZERO gradient right at the target. The old
    # version had a +/-2 cm deadzone (zero gradient in [0.38,0.42]) which let the CoM
    # sink ~2 cm for free before any penalty appeared -> no restoring signal at the
    # equilibrium. No deadzone now: the slope is non-zero from 0.40 downward, so
    # holding exactly 0.40 is strictly better than sinking. Still saturates (->1) so a
    # large error can never dominate / explode.
    #   err(m):  0.00 ->0.00 | 0.02 ->0.148 | 0.05 ->0.632 | 0.10 ->0.982
    err = body_height - self._config.reward_config.base_height_target
    return 1.0 - jp.exp(-jp.square(err / 0.05))

  def _cost_posture(self, qpos: jax.Array, command: jax.Array) -> jax.Array:
    # Neutral leg configuration, but ONLY when close to stationary: the gate
    # decays as the commanded speed grows, so dynamic leg motion during
    # locomotion is not penalized. This is the fix for zero-command leg spread:
    # the hip (highest weight) is pulled back to default whenever standing.
    err = (qpos - self._default_pose)[self._leg_ids]
    raw = jp.sum(self._posture_weights * jp.square(err))
    gate = jp.exp(
        -jp.sum(jp.square(command))
        / self._config.reward_config.posture_cmd_sigma
    )
    return raw * gate

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Dimensionless action difference (no 1/dt): the global x dt then leaves a
    # clean per-step magnitude instead of inverting to 1/dt.
    return jp.sum(jp.square(act - last_act))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    q = qpos[self._leg_ids]
    out = -jp.clip(q - self._soft_lowers, None, 0.0)
    out += jp.clip(q - self._soft_uppers, 0.0, None)
    return jp.sum(out)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  # --------------------------------------------------------------------
  # Commands and perturbations.
  # --------------------------------------------------------------------

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    cmd_shape = self._cmd_a.shape[0]
    y_k = jax.random.uniform(
        y_rng, shape=(cmd_shape,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(cmd_shape,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(cmd_shape,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      force = (
          u_t
          * self._torso_mass
          * state.info["pert_mag"]
          / state.info["pert_duration_seconds"]
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
      trigger = (
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"]
      )
      state.info["pert_steps"] = jp.where(trigger, 0, state.info["pert_steps"])
      state.info["pert_dir"] = jp.where(
          trigger, gen_dir(rng), state.info["pert_dir"]
      )
      return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )