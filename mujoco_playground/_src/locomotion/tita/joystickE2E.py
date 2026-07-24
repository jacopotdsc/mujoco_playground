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
      ctrl_dt=0.01,      # = sim_dt * decimation in Isaac
      sim_dt=0.002,
      episode_length=1000,
      # PD (Isaac: cfg.control.stiffness / damping).
      Kp=80.0,
      Kd=1.0,
      Kd_wheel=0.5,      # kv delle ruote (velocity control)
      action_repeat=1,
      # Isaac: cfg.control.action_scale_pos / action_scale_vel.
      action_scale_pos=0.5,
      action_scale_vel=10.0,
      # Isaac: cfg.normalization.clip_actions.
      clip_actions=100.0,
      soft_joint_pos_limit_factor=0.95,   # soft_dof_pos_limit
      soft_dof_vel_limit=1.0,
      soft_torque_limit=1.0,
      # Limiti di velocità dei giunti (Isaac li legge dall'URDF): 8 valori.
      dof_vel_limits=[20.0, 20.0, 20.0, 30.0, 20.0, 20.0, 20.0, 30.0],
      # Isaac: cfg.normalization.obs_scales.
      obs_scales=config_dict.create(
          lin_vel=2.0,
          ang_vel=0.25,
          dof_pos=1.0,
          dof_vel=0.05,
      ),
      # Isaac: cfg.noise.
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              dof_pos=0.01,
              dof_vel=1.5,
              gyro=0.2,       # ang_vel noise
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      # ------------------------------------------------------------------
      # SCALE DELLE REWARD: copia qui i valori dal tuo tita_config.py
      # (cfg.rewards.scales). Le scale a 0 disattivano il termine (come il
      # pop in _prepare_reward_function). Qui metto valori tipici.
      # ------------------------------------------------------------------
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base.
              lin_vel_z=-2.0,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              base_height=-10.0,
              inclination=-0.1,
              # Regularization.
              torques=-0.00001,
              dof_vel=-0.0,
              dof_acc=-2.5e-7,
              action_rate=-0.01,
              dof_pos_limits=-10.0,
              dof_vel_limits=-0.0,
              torque_limits=-0.0,
              # Contatti / piedi.
              collision=-1.0,
              stumble=-0.0,
              feet_air_time=1.0,
              no_fly=0.0,
              unbalance_feet_air_time=-0.0,
              unbalance_feet_height=-0.0,
              feet_contact_forces=-0.0,
              feet_distance=-100.0,
              feet_vel=-0.0,
              # Postura / simmetria (specifiche TITA).
              nominal_foot_position=2.0,
              leg_symmetry=1.0,
              same_foot_z_position=-5.0,
              same_foot_x_position=1.0,
              wheel_adjustment=1.0,
              stand_still=-1.0,
              # Terminazione / sopravvivenza.
              termination=-0.0,
              survival=0.0,
          ),
          only_positive_rewards=True,
          tracking_sigma=0.25,
          # Isaac: cfg.rewards.*
          base_height_target=0.35,
          foot_radius=0.0925,           # cfg.asset.foot_radius
          min_feet_air_time=0.25,
          max_feet_air_time=0.65,
          min_feet_distance=0.4,
          max_feet_distance=0.6,
          max_contact_force=300.0,
          nominal_foot_position_tracking_sigma=0.01,
          nominal_foot_position_tracking_sigma_wrt_v=0.5,
          leg_symmetry_tracking_sigma=0.001,
          foot_x_position_sigma=0.001,
      ),
      # Spinte casuali (Isaac: domain_rand.push_robots) — qui col meccanismo
      # a "velocity kick" di playground.
      pert_config=config_dict.create(
          enable=True,
          velocity_kick=[0.0, 1.0],     # ~ max_push_vel_xy
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      # Isaac: cfg.commands (ranges + resampling_time).
      command_config=config_dict.create(
          resampling_time=10.0,
          lin_vel_x=[-1.0, 1.0],
          lin_vel_y=[-0.0, 0.0],        # biped su ruote: y di solito 0
          ang_vel_yaw=[-1.0, 1.0],
          # I comandi piccoli vengono azzerati (come in Isaac).
          zero_command_threshold=0.2,
      ),
      impl="jax",
      naconmax=16 * 8192,
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
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._leg_ids = jp.array(consts.LEG_DOF_IDS)
    self._wheel_ids = jp.array(consts.WHEEL_DOF_IDS)

    # Soft joint limits — solo gambe (le ruote sono giunti continui).
    lowers, uppers = self.mj_model.jnt_range[1:].T
    c = (lowers + uppers) / 2
    r = uppers - lowers
    f = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = (c - 0.5 * r * f)[np.array(consts.LEG_DOF_IDS)]
    self._soft_uppers = (c + 0.5 * r * f)[np.array(consts.LEG_DOF_IDS)]

    self._torque_limits = jp.array(self.mj_model.actuator_forcerange[:, 1])
    self._dof_vel_limits = jp.array(self._config.dof_vel_limits)

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

    self._feet_touch_sensor_adr = jp.array([
        self._mj_model.sensor_adr[self._mj_model.sensor(name).id]
        for name in consts.FEET_TOUCH_SENSORS
    ])

    self._feet_floor_found_sensor = [
        self._mj_model.sensor(name).id
        for name in consts.FEET_FLOOR_FOUND_SENSORS
    ]

    cc = self._config.command_config
    self._cmd_lo = jp.array(
        [cc.lin_vel_x[0], cc.lin_vel_y[0], cc.ang_vel_yaw[0]]
    )
    self._cmd_hi = jp.array(
        [cc.lin_vel_x[1], cc.lin_vel_y[1], cc.ang_vel_yaw[1]]
    )
    self._steps_per_cmd = int(round(cc.resampling_time / self.dt))
    self._cmd_scale = jp.array([
        self._config.obs_scales.lin_vel,
        self._config.obs_scales.lin_vel,
        self._config.obs_scales.ang_vel,
    ])

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
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    #qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

    # Isaac _reset_dofs: dof_pos = default * U(0.5, 1.5), dof_vel = 0.
    rng, key = jax.random.split(rng)
    scale = jax.random.uniform(key, (consts.NUM_DOFS,), minval=0.5, maxval=1.5)
    #qpos = qpos.at[7:].set(self._default_pose * scale)

    # Isaac _reset_root_states: vel base U(-0.5, 0.5).
    rng, key = jax.random.split(rng)
    #qvel = qvel.at[0:6].set(
    #    jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    #)

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

    # Perturbazioni.
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key = jax.random.split(rng)
    cmd = self.sample_command(key)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "steps_until_next_cmd": self._steps_per_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "last_dof_vel": jp.zeros(consts.NUM_DOFS),
        "feet_air_time": jp.zeros(2),
        "last_feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),            # current_max_feet_height
        "last_max_feet_height": jp.zeros(2),
        "steps_until_next_pert": jp.round(
            time_until_next_pert / self.dt
        ).astype(jp.int32),
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": jp.round(pert_duration_seconds / self.dt).astype(
            jp.int32
        ),
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info, jp.zeros(self.mjx_model.nu))
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)

    action = jp.clip(
        action, -self._config.clip_actions, self._config.clip_actions
    )

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

    obs = self._get_obs(data, state.info, action)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    # Isaac: only_positive_rewards clippa la somma (senza termination),
    # poi aggiunge la termination dopo il clip.
    reward = sum(v for k, v in rewards.items() if k != "termination") * self.dt
    if self._config.reward_config.only_positive_rewards:
      reward = jp.clip(reward, 0.0, 10000.0)
    reward += rewards["termination"] * self.dt

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
    state.info["rng"], key = jax.random.split(state.info["rng"])
    resample = done | (state.info["steps_until_next_cmd"] <= 0)
    state.info["command"] = jp.where(
        resample, self.sample_command(key), state.info["command"]
    )
    state.info["steps_until_next_cmd"] = jp.where(
        resample, self._steps_per_cmd, state.info["steps_until_next_cmd"]
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
    scales = self._config.obs_scales

    gyro = self.get_gyro(data)
    gravity = self.get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]

    def add_noise(x, rng_key, scale):
      return x + (2 * jax.random.uniform(rng_key, shape=x.shape) - 1) * (
          noise.level * scale
      )

    info["rng"], k1, k2, k3, k4 = jax.random.split(info["rng"], 5)
    noisy_gyro = add_noise(gyro, k1, noise.scales.gyro)
    noisy_gravity = add_noise(gravity, k2, noise.scales.gravity)
    noisy_joint_angles = add_noise(joint_angles, k3, noise.scales.dof_pos)
    noisy_joint_vel = add_noise(joint_vel, k4, noise.scales.dof_vel)

    # Stessa composizione di Isaac: niente lin_vel, niente pos delle ruote.
    leg_pos_err = (noisy_joint_angles - self._default_pose)[self._leg_ids]
    state = jp.hstack([
        noisy_gyro * scales.ang_vel,          # 3
        noisy_gravity,                        # 3
        leg_pos_err * scales.dof_pos,         # 6
        noisy_joint_vel * scales.dof_vel,     # 8
        action,                               # 8
        info["command"] * self._cmd_scale,    # 3
    ])  # tot: 31

    # Privileged: versione pulita + extra (linvel vera, forze, contatti...).
    linvel = self.get_local_linvel(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,                                             # 3
        self.get_accelerometer(data),                     # 3
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

    return {"state": state, "privileged_state": privileged_state}

  # --------------------------------------------------------------------
  # Helper geometrici.
  # --------------------------------------------------------------------

  def _feet_pos_base_frame(self, data: mjx.Data) -> jax.Array:
    """Posizione dei piedi nel frame base, shape (2, 3)."""
    base_pos = data.xpos[self._torso_body_id]
    r_base = data.xmat[self._torso_body_id].reshape(3, 3)
    p_rel = data.site_xpos[self._feet_site_id] - base_pos
    return p_rel @ r_base  # == (R^T @ p_rel^T)^T

  # --------------------------------------------------------------------
  # Rewards (migrate 1:1 da no_constrains_legged_robot.py).
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
    local_linvel = self.get_local_linvel(data)
    gyro = self.get_gyro(data)
    gravity = self.get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    torques = data.actuator_force
    cmd = info["command"]
    feet_pos_b = self._feet_pos_base_frame(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]  # (2, 3) world

    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(cmd, local_linvel),
        "tracking_ang_vel": self._reward_tracking_ang_vel(cmd, gyro),
        "lin_vel_z": self._reward_lin_vel_z(local_linvel),
        "ang_vel_xy": self._reward_ang_vel_xy(gyro),
        "orientation": self._reward_orientation(gravity),
        "base_height": self._reward_base_height(data),
        "inclination": self._reward_inclination(gyro),
        "torques": self._reward_torques(torques),
        "dof_vel": self._reward_dof_vel(joint_vel),
        "dof_acc": self._reward_dof_acc(joint_vel, info["last_dof_vel"]),
        "action_rate": self._reward_action_rate(action, info["last_act"]),
        "dof_pos_limits": self._reward_dof_pos_limits(joint_angles),
        "dof_vel_limits": self._reward_dof_vel_limits(joint_vel),
        "torque_limits": self._reward_torque_limits(torques),
        "collision": self._reward_collision(data),
        "stumble": self._reward_stumble(data, contact),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact
        ),
        "no_fly": self._reward_no_fly(contact),
        "unbalance_feet_air_time": jp.var(info["last_feet_air_time"]),
        "unbalance_feet_height": jp.var(info["last_max_feet_height"]),
        "feet_contact_forces": self._reward_feet_contact_forces(data),
        "feet_distance": self._reward_feet_distance(data),
        "feet_vel": self._reward_feet_vel(feet_vel),
        "nominal_foot_position": self._reward_nominal_foot_position(
            feet_pos_b, cmd
        ),
        "leg_symmetry": self._reward_leg_symmetry(feet_pos_b),
        "same_foot_z_position": jp.square(feet_pos_b[0, 2] - feet_pos_b[1, 2]),
        "same_foot_x_position": self._reward_same_foot_x_position(feet_pos_b),
        "wheel_adjustment": self._reward_wheel_adjustment(gravity, feet_vel),
        "stand_still": self._reward_stand_still(joint_angles, cmd),
        "termination": done,
        "survival": (info["step"] * self.dt > 10.0),
    }

  # Tracking.

  def _reward_tracking_lin_vel(self, cmd, local_vel) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(cmd[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(self, cmd, gyro) -> jax.Array:
    ang_vel_error = jp.square(cmd[2] - gyro[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base.

  def _reward_lin_vel_z(self, local_linvel) -> jax.Array:
    # Isaac usa base_lin_vel (frame base).
    return jp.square(local_linvel[2])

  def _reward_ang_vel_xy(self, gyro) -> jax.Array:
    return jp.sum(jp.square(gyro[:2]))

  def _reward_orientation(self, gravity) -> jax.Array:
    # Versione TITA: booleana, norm(g_xy) > 0.1.
    return (jp.linalg.norm(gravity[:2]) > 0.1)

  def _reward_base_height(self, data: mjx.Data) -> jax.Array:
    # Penalizza solo se la base è SOTTO il target (clip in [-1, 0]).
    base_height = data.qpos[2]  # su flat terrain
    err = base_height - self._config.reward_config.base_height_target
    return jp.abs(jp.clip(err, -1.0, 0.0))

  def _reward_inclination(self, gyro) -> jax.Array:
    # Penalizza vel. angolare di roll/pitch.
    return jp.linalg.norm(gyro[:2])

  # Regularization.

  def _reward_torques(self, torques) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _reward_dof_vel(self, joint_vel) -> jax.Array:
    return jp.sum(jp.square(joint_vel))

  def _reward_dof_acc(self, joint_vel, last_joint_vel) -> jax.Array:
    return jp.sum(jp.square((last_joint_vel - joint_vel) / self.dt))

  def _reward_action_rate(self, act, last_act) -> jax.Array:
    return jp.sum(jp.square(last_act - act))

  def _reward_dof_pos_limits(self, joint_angles) -> jax.Array:
    # Solo gambe (le ruote non hanno limiti di posizione).
    q = joint_angles[self._leg_ids]
    out_of_limits = -jp.clip(q - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(q - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _reward_dof_vel_limits(self, joint_vel) -> jax.Array:
    soft = self._config.soft_dof_vel_limit
    return jp.sum(
        jp.clip(jp.abs(joint_vel) - self._dof_vel_limits * soft, 0.0, 1.0)
    )

  def _reward_torque_limits(self, torques) -> jax.Array:
    soft = self._config.soft_torque_limit
    return jp.sum(jp.clip(jp.abs(torques) - self._torque_limits * soft, 0.0, None))

  # Contatti / piedi.

  def _reward_collision(self, data: mjx.Data) -> jax.Array:
    # Contatti su body penalizzati (es. cosce) con il pavimento.
    return jp.sum(
        jp.array([
            geoms_colliding(data, gid, self._floor_geom_id)
            for gid in self._collision_geom_id
        ])
    )

  def _reward_stumble(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    # Isaac: |F_xy| > 5*|F_z| sui piedi. In MJX le forze di contatto per-geom
    # sono costose: approssimiamo con vel. tangenziale alta a contatto.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = jp.linalg.norm(feet_vel[..., :2], axis=-1)
    return jp.any((vel_xy > 1.0) & contact)

  def _reward_feet_air_time(self, air_time, first_contact) -> jax.Array:
    # Versione TITA: premia durate del passo tra [min, max].
    rc = self._config.reward_config
    below = jp.sum(
        jp.minimum(air_time - rc.min_feet_air_time, 0.0) * first_contact
    )
    above = jp.sum(
        jp.minimum(rc.max_feet_air_time - air_time, 0.0) * first_contact
    )
    return below + above

  def _reward_no_fly(self, contact) -> jax.Array:
    single_contact = jp.sum(contact) == 1
    return single_contact

  def _reward_feet_contact_forces(self, data: mjx.Data) -> jax.Array:
    forces = data.sensordata[self._feet_touch_sensor_adr]
    return jp.sum(
        jp.clip(forces - self._config.reward_config.max_contact_force, 0.0, None)
    )

  def _reward_feet_distance(self, data: mjx.Data) -> jax.Array:
    rc = self._config.reward_config
    p = data.site_xpos[self._feet_site_id]
    d = jp.linalg.norm(p[0, :2] - p[1, :2])
    return jp.clip(rc.min_feet_distance - d, 0.0, 1.0) + jp.clip(
        d - rc.max_feet_distance, 0.0, 1.0
    )

  def _reward_feet_vel(self, feet_vel) -> jax.Array:
    return jp.linalg.norm(feet_vel[0]) + jp.linalg.norm(feet_vel[1])

  # Postura / simmetria (specifiche del biped su ruote).

  def _reward_nominal_foot_position(self, feet_pos_b, cmd) -> jax.Array:
    rc = self._config.reward_config
    nominal_z = -(rc.base_height_target - rc.foot_radius)
    height_error = nominal_z - feet_pos_b[:, 2]
    reward = jp.mean(
        jp.exp(-jp.square(height_error) / rc.nominal_foot_position_tracking_sigma)
    )
    vel_cmd_norm = jp.linalg.norm(cmd)
    return reward * jp.exp(
        -jp.square(vel_cmd_norm) / rc.nominal_foot_position_tracking_sigma_wrt_v
    )

  def _reward_leg_symmetry(self, feet_pos_b) -> jax.Array:
    err = jp.abs(feet_pos_b[0, 1]) - jp.abs(feet_pos_b[1, 1])
    return jp.exp(
        -jp.square(err) / self._config.reward_config.leg_symmetry_tracking_sigma
    )

  def _reward_same_foot_x_position(self, feet_pos_b) -> jax.Array:
    err = feet_pos_b[0, 0] - feet_pos_b[1, 0]
    return jp.exp(
        -jp.square(err) / self._config.reward_config.foot_x_position_sigma
    )

  def _reward_wheel_adjustment(self, gravity, feet_vel) -> jax.Array:
    # Premia lo slittamento delle ruote nella direzione dell'inclinazione.
    incline_x = gravity[0]
    wheel_x_mean = (feet_vel[0, 0] + feet_vel[1, 0]) / 2.0
    invalid = (feet_vel[0, 0] * feet_vel[1, 0]) < 0
    wheel_x_mean = jp.where(invalid, 0.0, wheel_x_mean)
    return (incline_x * wheel_x_mean > 0)

  def _reward_stand_still(self, joint_angles, cmd) -> jax.Array:
    # Penalizza deviazione dei giunti gamba quando il comando vx ~ 0
    # (le ruote sono escluse, come in Isaac).
    dof_err = (joint_angles - self._default_pose)[self._leg_ids]
    return jp.sum(jp.abs(dof_err)) * (jp.abs(cmd[0]) < 0.05)

  # --------------------------------------------------------------------
  # Comandi e perturbazioni.
  # --------------------------------------------------------------------

  def sample_command(self, rng: jax.Array) -> jax.Array:
    cmd = jax.random.uniform(
        rng, shape=(3,), minval=self._cmd_lo, maxval=self._cmd_hi
    )
    # Azzera i comandi piccoli (Isaac: norm(cmd_xy) <= 0.2 -> 0).
    keep = jp.linalg.norm(cmd[:2]) > (
        self._config.command_config.zero_command_threshold
    )
    return cmd.at[:2].set(cmd[:2] * keep)

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