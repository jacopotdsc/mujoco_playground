# Joystick task for TITA (wheeled biped), ported from the Isaac Gym
# implementation (no_constrains_legged_robot.py) to mujoco_playground / MJX.
"""Joystick task for TITA."""

from typing import Any, Dict, Optional, Union, Tuple

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.tita import base as tita_base
from mujoco_playground._src.locomotion.tita import tita_constants as consts

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
      Kp=35.0,
      Kd=10.0,
      Kd_wheel=0.5,      # kv delle ruote (velocity control)
      action_repeat=1,
      action_scale_pos=0.5,
      action_scale_vel=15.0,
      soft_joint_pos_limit_factor=0.95, 
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,       # ang_vel noise
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              action_rate_first_order=-0.000,
              action_rate_second_order=-0.0000,
              torques=-0.0001,
              orientation=-1.0,
              base_height=-2.0,  # cost: saturating penalty, see _cost_height
              joint_regularization=-1.0,
              termination=-2.0,
              lin_vel_z=-0.1,
              ang_vel_xy=-0.3,
              dof_pos_limits=-0.1,
              energy=-0.0001,
              wheel_track=-2.0,
          ),
          # Come nel file MPC: somma con segno, clip simmetrico (niente
          # only_positive_rewards alla Isaac).
          only_positive_rewards=False,
          tracking_sigma=0.25,
          max_foot_height=0.1,
          base_height_target=0.40,
          wheel_track_target=0.567,  # nominal distance between the two wheels (m)
      ),
       pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      # Isaac: cfg.commands (ranges + resampling_time).
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[1.0, 0.5],
          # Probability of not zeroing out new command.
          b=[0.9, 0.5],
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

    print("biastype ", self._mj_model.actuator_biastype)   # 0 = motor puro; !=0 = affine (position)
    print("gainprm0 ", self._mj_model.actuator_gainprm[:, 0])
    print("biasprm1 ", self._mj_model.actuator_biasprm[:, 1])
    print("biasprm2", self._mj_model.actuator_biasprm[:, 2])
    print("ctrllimited", self._mj_model.actuator_ctrllimited)   # deve essere tutto 0
    print("ctrlrange", self._mj_model.actuator_ctrlrange)

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._leg_ids = jp.array(consts.LEG_DOF_IDS)
    self._wheel_ids = jp.array(consts.WHEEL_DOF_IDS)

    # Soft joint limits — solo gambe (le ruote sono giunti continui).
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers[self._leg_ids] * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers[self._leg_ids] * self._config.soft_joint_pos_limit_factor

    self._torque_limits = jp.array(self.mj_model.actuator_forcerange[:, 1])

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom(consts.FLOOR_GEOM).id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    self._termination_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.TERMINATION_GEOMS]
    )
    self._collision_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.COLLISION_GEOMS]
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

    self._base_com_adr = self._sensor_adr("base_subtree_com")

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

  # --------------------------------------------------------------------
  # Reset / step.
  # --------------------------------------------------------------------

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x,y = +U(-0.5, 0.5), yaw = U(-pi, pi).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

    # Solo le velocità lineari orizzontali (vx, vy) sono randomizzate.
    # vz e le velocità angolari (roll/pitch/yaw rate) restano a zero per
    # non iniettare una penalità istantanea (lin_vel_z, ang_vel_xy)
    # indipendente dalla policy fin dal primo step dell'episodio.
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:2].set(
        jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    )

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

    '''
    wheel_ids = jp.array(consts.WHEEL_DOF_IDS)
    def substep_fn(data, _):
        q = data.qpos[7:]
        dq = data.qvel[6:]

        # --- Gambe: PD di posizione (P_AND_V di Isaac) ---
        q_des = self._default_pose + action * self._config.action_scale_pos
        tau_p = self._config.Kp * (q_des - q)
        tau_p = tau_p.at[wheel_ids].set(0.0)          # niente P sulle ruote

        # --- Termine D: Kd sulle gambe (target vel = 0),
        #     Kd_wheel sulle ruote (target vel = action * action_scale_vel) ---
        dq_des = jp.zeros_like(dq).at[wheel_ids].set(
            action[wheel_ids] * self._config.action_scale_vel
        )
        kd = jp.full_like(dq, self._config.Kd).at[wheel_ids].set(
            self._config.Kd_wheel
        )
        tau_d = kd * (dq_des - dq)

        # Clip dei torque (Isaac: clip(torques, ±torque_limits)).
        tau = jp.clip(tau_p + tau_d, -self._torque_limits, self._torque_limits)

        data = data.replace(ctrl=tau)
        data = mjx.step(self.mjx_model, data)

        llc_log = {
            "q_des": q_des,
            "dq_des": dq_des,
            "tau_p": tau_p,
            "tau_d": tau_d,
            "tau": tau,
            "action": action,
            "qpos": data.qpos,
        }
        return data, llc_log
    
    data, llc_logs = jax.lax.scan(substep_fn, state.data, None, length=self.n_substeps)
    state = state.replace(data=data)
    '''

    # Stati dei piedi (equivalente di _compute_feet_states).
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
    state = jp.hstack([
        noisy_linvel,   # 3
        noisy_gyro, # 3
        noisy_gravity,  # 3
        leg_pos_err,    # 6
        noisy_joint_vel,    # 8
        action, # 8
        info["command"] ,  # 3
    ])  # tot: 31

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,                                             # 3
        accelerometer,                     # 3
        gravity,                                          # 3
        linvel,                                           # 3
        angvel,                                           # 3
        (joint_angles - self._default_pose)[self._leg_ids],  # 6
        joint_vel,                                        # 8
        data.actuator_force,                              # 8
        info["last_contact"],                             # 2
        feet_vel,                                         # 2*3
        info["feet_air_time"],                            # 2
        data.qpos[2:3],                                   # base height, 1
        data.xfrc_applied[self._torso_body_id, :3],       # push force, 3
    ])

    return {
        "state": state, 
        "privileged_state": privileged_state
    }

  # --------------------------------------------------------------------
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
    del first_contact, contact  # Non usati in questo set di reward.

    body_height = data.sensordata[self._base_com_adr][2]

    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "action_rate_first_order": self._cost_action_rate_first_order(
            action, info["last_act"]
        ),
        "action_rate_second_order": self._cost_action_rate_second_order(
            action, info["last_act"], info["last_last_act"]
        ),
        "torques": self._cost_torques(data.actuator_force),
        "orientation": self._cost_orientation(data),
        "base_height": self._cost_height(body_height),
        "joint_regularization": self._cost_joint_regularization(
            data.qpos[7:]
        ),
        "termination": self._cost_termination(done),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "wheel_track": self._cost_wheel_track(data),
    }

  # Tracking (versione con errore normalizzato, come nel file MPC).

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    # Stessa struttura di kernel gaussiano di _reward_tracking_ang_vel:
    # errore grezzo (non normalizzato per il comando), altrimenti a
    # comandi alti basta restare fermi per ottenere reward parziale.
    term_error = commands[0] - local_vel[0]
    term_norm = jp.square(term_error)
    return jp.exp(-term_norm / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, ang_vel: jax.Array
  ) -> jax.Array:
    term_error = commands[-1] - ang_vel[2]
    term_norm = jp.square(term_error)
    return jp.exp(-term_norm / self._config.reward_config.tracking_sigma)

  # Base.

  def _cost_orientation(self, data: mjx.Data) -> jax.Array:
    # Somma dei quadrati delle componenti x/y del gravity vector: 0 quando
    # la base è perfettamente piana, cresce quanto più si inclina.
    gravity_xy = self.get_gravity(data)[:2]
    return jp.sum(jp.square(gravity_xy))

  def _cost_height(self, body_height: jax.Array) -> jax.Array:
    # Deadzone di 2cm (nessuna penalità entro lo scostamento fisiologico),
    # poi saturazione esponenziale: l'errore non esplode mai, tende
    # asintoticamente a 1 invece di crescere senza limite come il
    # precedente termine quadratico puro.
    h_error = jp.abs(
        body_height - self._config.reward_config.base_height_target
    )
    deadzone_error = jp.maximum(0.0, h_error - 0.02)
    return 1.0 - jp.exp(-jp.square(deadzone_error / 0.03))

  def _cost_wheel_track(self, data: mjx.Data) -> jax.Array:
    # Penalize solo la componente orizzontale (xy) della distanza tra i
    # piedi: l'offset verticale è lasciato libero, così le gambe possono
    # sfalsarsi per sterzare o assorbire urti senza essere penalizzate.
    feet_pos = data.site_xpos[self._feet_site_id]
    dist = jp.linalg.norm(feet_pos[0, :2] - feet_pos[1, :2])
    return jp.square(dist - self._config.reward_config.wheel_track_target)

  def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  # Energia / regolarizzazione.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate_first_order(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    term_error = (act - last_act) / self._config.ctrl_dt
    return jp.sum(jp.square(term_error))

  def _cost_action_rate_second_order(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    term_error = (act - 2 * last_act + last_last_act) / self._config.ctrl_dt
    return jp.sum(jp.square(term_error))

  def _cost_joint_regularization(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose (ruote escluse via peso 0).
    weight = jp.array([1.0, 1.0, 1.0, 0.0] * 2)
    scale = 1.0 / (self._mj_model.nu - 2)
    return scale * jp.sum(jp.square(qpos - self._default_pose) * weight)

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Solo gambe (le ruote non hanno limiti di posizione).
    q = qpos[self._leg_ids]
    out_of_limits = -jp.clip(q - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(q - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  # --------------------------------------------------------------------
  # Comandi e perturbazioni.
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