# Joystick task for TITA (two-wheeled balancing biped) on MuJoCo Playground / MJX.
#
# The robot has two 3-DOF legs, each ending in a driven wheel:
#   qpos[7:] index -> 0,1,2 = left  hip/thigh/knee, 3 = left  wheel
#                     4,5,6 = right hip/thigh/knee, 7 = right wheel
#   LEG_DOF_IDS = [0,1,2,4,5,6]   WHEEL_DOF_IDS = [3,7]
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

import mpx.config.config_dfcip as config
from mpx.utils.mpc_wrapper_dfcip import BatchedMPCControllerWrapper
import mpx.utils.sim as sim_utils
import mpx.utils.mpc_utils as mpc_utils


def _tree_has_nonfinite(tree) -> jax.Array:
  """True if any float leaf of the pytree contains NaN/Inf."""
  leaves = jax.tree_util.tree_leaves(tree)
  checks = [
      jp.any(~jp.isfinite(x))
      for x in leaves
      if jp.issubdtype(jp.asarray(x).dtype, jp.floating)
  ]
  if not checks:
    return jp.array(False)
  return jp.stack(checks).any()


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
      Kd_wheel=0.5,
      action_repeat=1,
      action_scale_pos=0.5, # prova 0.3
      action_scale_vel=25.0, # prova 15.0
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
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              tracking_mpc_accel=0.0,
              tracking_mpc_alpha=0.0,
              orientation=-1.0,
              ang_vel_xy=-0.3,
              base_height=-1.0,
              posture=-1.0,      # command-gated; see _cost_posture
              torques=-1e-4,
              action_rate=-0.01,
              dof_pos_limits=-1.0,
              dof_vel=-0.0,
              termination=-100.0,
          ),
          only_positive_rewards=False,
          tracking_sigma=0.0625,
          mpc_tracking_sigma=0.0625,
          base_height_target=0.4, 
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
          h=[0.4, 0.4],    # CoM height command range [min, max] [m]
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
    self._cmd_h = jp.array(self._config.command_config.h)
    self._cmd_resample_scale = 0.5 * self._config.episode_length * self.dt

    self._posture_weights = jp.array([1.0, 0.5, 0.5, 1.0, 0.5, 0.5])

    self._base_com_linvel_adr = self._sensor_adr("base_subtree_linvel")

    # Batched MPC/WBC wrapper (batch size 1), same construction as the MPC env.
    sim_frequency = 1.0 / self.sim_dt
    self.mpc_period = max(1, int(sim_frequency / config.mpc_frequency))
    self.mpc = BatchedMPCControllerWrapper(config, 1)
    self._wheel_radius = jp.array([self._mj_model.geom(name).size[0] for name in consts.FEET_GEOMS])
    self._wheel_base = self.mpc.config.d

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

  def _joint_targets_from_qddot(self, qpos_joint, qvel_joint, qddot):
    """Desired joint pos/vel = current state integrated forward with the WBC's
    solved joint accelerations. qddot is (1, nv) from the vmapped solver
    (batch dim kept, unlike mpc_tau): [:, :6] is the floating base, [:, 6:] the
    nj actuated joints in qpos[7:]/qvel[6:] order."""
    dt = self._config.sim_dt
    qddot_joint = qddot[0, 6:]
    dq_target = qvel_joint + qddot_joint * dt
    q_target = qpos_joint + qvel_joint * dt + 0.5 * qddot_joint * dt**2
    return q_target, dq_target

  def _compute_joint_desired(self, action, qpos_joint, qvel_joint, qddot):
    #q_target, dq_target = self._joint_targets_from_qddot(qpos_joint, qvel_joint, qddot)
    
    q_target = self._default_pose
    dq_target = jp.zeros_like(action)

    q_des = q_target + action * self._config.action_scale_pos
    dq_des = dq_target + action * self._config.action_scale_vel

    return q_des, dq_des

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
    rng, key = jax.random.split(rng)
    joint_vel = jax.random.uniform(key, shape=qvel[6:].shape, minval=-0.2, maxval=0.2,)
    #qvel = qvel.at[6:].set(joint_vel)

    ctrl = jp.zeros(self.mjx_model.nu)

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
    time_until_next_cmd = jax.random.exponential(key1) * self._cmd_resample_scale
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    cmd = jax.random.uniform(
        key2, shape=(self._cmd_a.shape[0],), minval=-self._cmd_a, maxval=self._cmd_a
    )

    _, height_rng = jax.random.split(rng)
    base_height_target = self.sample_height(height_rng)

    # ------------------------------------------------------------------
    # MPC augmentation: initialise the persistent MPC state and run one
    # passive pass so all MPC info fields are populated with correct shapes.
    # The command fed to the MPC starts at zero (matches info["command"]).
    # ------------------------------------------------------------------
    mpc_state = self.mpc.init_state()
    mpc_state, tita_state, dfcip_state, mpc_tau, mpc_qddot, mpc_fl, mpc_fr, desired, theta_prev, mpc_reference = self._run_mpc_wbc(
        data=data,
        qpos=data.qpos,
        qvel=data.qvel,
        command=cmd,
        base_height_target=base_height_target,
        action=jp.zeros(self.mjx_model.nu),
        mpc_state=mpc_state,
        theta_prev=0.0,
        timestep=0
        )

    q_des, dq_des = self._compute_joint_desired(
        action=jp.zeros(self.mjx_model.nu),
        qpos_joint=data.qpos[7:],
        qvel_joint=data.qvel[6:],
        qddot=mpc_qddot,
    )

    info = {
        "rng": rng,
        "step": 0,
        "command": jp.zeros_like(cmd),
        "base_height_target": base_height_target,
        "target_command" : cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "last_dof_vel": jp.zeros(consts.NUM_DOFS),
        "last_local_linvel": self.get_local_linvel(data),
        "last_gyro": self.get_gyro(data),
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
        # --- MPC augmentation: persistent state + logged outputs. ---
        # These are updated every step() by a passive MPC/WBC pass and never
        # influence the actuator commands (see step()).
        "mpc_state": mpc_state,       # persistent MPC/WBC internal state (pytree)
        "mpc_state_last": mpc_state,  # previous-step MPC/WBC state, for delta obs (0 on reset)
        "theta_prev": theta_prev,     # unwrapped yaw carried across steps
        "tita_state": tita_state,     # raw 18-vec [pcom,vcom,pl,pr,dpl,dpr] (LOGGED ONLY)
        "dfcip_state": dfcip_state,   # DFCIP state x0 (feeds plot_command_tracking)
        "dfcip_state_last": dfcip_state,  # previous-step DFCIP state, for delta obs (0 on reset)
        "mpc_tau": mpc_tau,           # WBC feedforward torque (LOGGED ONLY)
        "mpc_qddot": mpc_qddot,       # WBC joint accelerations (LOGGED ONLY)
        "mpc_grf_left": mpc_fl,       # left-wheel ground reaction force
        "mpc_grf_right": mpc_fr,      # right-wheel ground reaction force
        # MPC control: first-stage optimal control u_ref, from mpc.run()'s reference.
        "mpc_control": mpc_reference[0][0, 13:],
        "mpc_control_last": mpc_reference[0][0, 13:],  # previous-step u_ref, for delta obs (0 on reset)
        # WBC desired vector (com/wheel/base/joint refs); obs slices out q/wheel-vel desired.
        "mpc_desired": desired[0],
        "joint_pos_des": q_des,  # desired leg joint positions (integrated from WBC qddot)
        "wheel_vel_des": dq_des,  # desired wheel velocities (integrated from WBC qddot)
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

  def _run_mpc_wbc(self, data, qpos: jax.Array, qvel: jax.Array, command: jax.Array, base_height_target, action, mpc_state, theta_prev, timestep: int):
    """Run the MPC planner (called every mpc_period steps)."""
    qpos = jp.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
    qvel = jp.nan_to_num(qvel, nan=0.0, posinf=0.0, neginf=0.0)

    contact_ids = sim_utils.geom_ids(self._mj_model, config.contact_frame)
    
    tita_state = self.build_tita_state(data)
    x0, theta_prev = self.get_dfip_current_state(tita_state, theta_prev)

    mpc_command = jp.array([command[0], 0.0, command[1], base_height_target])
    mpc_state, reference = self.mpc.run(mpc_state, x0[None, :], mpc_command[None, :])

    pl_world_wbc = tita_state[6:9][None, :]
    pr_world_wbc = tita_state[9:12][None, :]
    dpl_world_wbc = tita_state[12:15][None, :]
    dpr_world_wbc = tita_state[15:18][None, :]

    joint_scale = 0.1
    wheel_scale = 1.0
    scaled_weights = jp.array([joint_scale, joint_scale, joint_scale, wheel_scale]*2)
    scaled_action = action #* scaled_weights

    mpc_state, tau, qddot, fl, fr, desired = self.mpc.whole_body_run(
        mpc_state,
        x0,
        jp.asarray(qpos)[None, :],
        jp.asarray(qvel)[None, :],
        pl_world_wbc,
        pr_world_wbc,
        dpl_world_wbc,
        dpr_world_wbc,
        #scaled_action[None, :],
        use_nn=False
    )

    tau = jp.nan_to_num(tau[0], nan=0.0, posinf=0.0, neginf=0.0)
    qddot = jp.nan_to_num(qddot, nan=0.0, posinf=0.0, neginf=0.0)
    return mpc_state, tita_state, x0, tau, qddot, fl, fr, desired, theta_prev, reference

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    
    new_mpc_state, tita_state, dfcip_state, new_tau, new_qddot, mpc_fl, mpc_fr, desired, theta_prev, mpc_reference = self._run_mpc_wbc(
        data=state.data,
        qpos=state.data.qpos,
        qvel=state.data.qvel,
        command=state.info["command"],
        base_height_target=state.info["base_height_target"],
        action=action,
        mpc_state=state.info["mpc_state"],
        theta_prev=state.info["theta_prev"],
        timestep=state.info["step"]
    )
        
    bad = (
        _tree_has_nonfinite(new_mpc_state)
        | jp.any(~jp.isfinite(new_tau))
        | jp.any(~jp.isfinite(new_qddot))
    )
    mpc_state = jax.tree_util.tree_map(
        lambda new, old: jp.where(bad, old, new),
        new_mpc_state,
        state.info["mpc_state"],
    )
    tau   = jp.where(bad, state.info["mpc_tau"],   new_tau)
    qddot = jp.where(bad, state.info["mpc_qddot"], new_qddot)


    state.info["dfcip_state_last"] = state.info["dfcip_state"]
    state.info["mpc_control_last"] = state.info["mpc_control"]
    state.info["mpc_state_last"] = state.info["mpc_state"]

    state.info["mpc_state"]     = mpc_state
    state.info["mpc_tau"]       = tau
    state.info["mpc_qddot"]     = qddot
    state.info["theta_prev"]    = theta_prev
    state.info["tita_state"]    = tita_state
    state.info["dfcip_state"]   = dfcip_state
    state.info["mpc_grf_left"]  = mpc_fl
    state.info["mpc_grf_right"] = mpc_fr
    state.info["mpc_control"]   = mpc_reference[0][0, 13:]
    state.info["mpc_desired"]   = desired[0]

    #q_des = self._default_pose + action * self._config.action_scale_pos
    #v_des = action * self._config.action_scale_vel

    # Computed here (pre-substep qpos/qvel, same instant as the qddot above),
    # written to state.info only after the substep loop below.

    q_des, dq_des = self._compute_joint_desired(
        action=action,
        qpos_joint=state.data.qpos[7:],
        qvel_joint=state.data.qvel[6:],
        qddot=qddot,
    )

    def substep_fn(data, _):
        q  = data.qpos[7:]
        qd = data.qvel[6:]

        #q_des, dq_des = self._compute_joint_desired(
        #    action=action,
        #    qpos_joint=state.data.qpos[7:],
        #    qvel_joint=state.data.qvel[6:],
        #    qddot=qddot,
        #)

        tau_leg   = self._config.Kp * (q_des - q) - self._config.Kd * qd
        tau_wheel = self._config.Kd_wheel * (dq_des - qd)

        ctrl_nn = jp.zeros(self.mjx_model.nu)
        ctrl_nn = ctrl_nn.at[self._leg_ids].set(tau_leg[self._leg_ids])
        ctrl_nn = ctrl_nn.at[self._wheel_ids].set(tau_wheel[self._wheel_ids])

        ctrl = ctrl_nn + tau

        data = data.replace(ctrl=ctrl)
        data = mjx.step(self.mjx_model, data)
        return data, None

    data, _ = jax.lax.scan(substep_fn, state.data, xs=None, length=self.n_substeps)
    state = state.replace(data=data)

    state.info["joint_pos_des"] = q_des
    state.info["wheel_vel_des"] = dq_des

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

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    # Aggiorna i buffer (equivalente della coda di post_physics_step).
    state.info["step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_dof_vel"] = data.qvel[6:]
    state.info["last_local_linvel"] = self.get_local_linvel(data)   
    state.info["last_gyro"] = self.get_gyro(data)                   
    state.info["feet_air_time"] *= ~contact_filt
    state.info["swing_peak"] *= ~contact_filt
    state.info["last_contact"] = contact

    # Ricampionamento comandi a intervallo fisso (Isaac: resampling_time).
    state.info["steps_until_next_cmd"] -= 0
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["target_command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["target_command"]),
        state.info["target_command"],
    )
    state.info["command"] = (
        state.info["command"]
        + 0.02 * (state.info["target_command"] - state.info["command"])
    )
    state.info["steps_until_next_cmd"] = jp.where(
        (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * self._cmd_resample_scale / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )

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

    # CoM-height error (com_z - target). The actor was previously BLIND to height:
    # height only lived in privileged_state (critic). Without this signal the policy
    # cannot close a loop on CoM_z and slowly sinks. Centred near 0 so the running
    # obs-normalizer behaves. Same CoM sensor used by the height reward.
    current_com_height = data.sensordata[self._base_com_adr][2]
    com_height_err = (current_com_height - info["base_height_target"])

    # MPC augmentation: DFCIP/LIP state + MPC control, so the policy sees what
    # the MPC/WBC pipeline is planning alongside its own proprioception.
    # DFCIP state layout (see get_dfip_current_state / reference_generator_dfcip_online):
    #   x = [pcom(3), vcom(3), c(3), vcontact_z(1), theta(1), v(1), omega(1)]
    # Delta vs previous step (0 right after reset) instead of the raw absolute
    # state, since pcom/c drift unbounded in world frame.
    dfcip_state = info["dfcip_state"]
    dfcip_state_last = info["dfcip_state_last"] 
    dfcip_pcom  = dfcip_state[0:3]   - dfcip_state_last[0:3]      # CoM position delta
    dfcip_vcom  = dfcip_state[3:6]   - dfcip_state_last[3:6]      # CoM velocity delta
    dfcip_c     = dfcip_state[6:9]   - dfcip_state_last[6:9]      # contact-point (wheel midpoint) position delta
    dfcip_vc_z  = dfcip_state[9:10]  #- dfcip_state_last[9:10]     # contact-point vertical velocity delta
    dfcip_theta = dfcip_state[10:11] #- dfcip_state_last[10:11]    # base yaw delta
    dfcip_v     = dfcip_state[11:12] - dfcip_state_last[11:12]    # forward velocity delta
    dfcip_omega = dfcip_state[12:13] #- dfcip_state_last[12:13]    # yaw rate delta

    # MPC control: first-stage optimal control u_ref = [a, ac_z, alpha, Fl(3), Fr(3)].
    # Delta vs previous step (0 right after reset), same treatment as dfcip_state.
    mpc_control = info["mpc_control"]
    mpc_control_last = info["mpc_control_last"]  
    mpc_control_a = mpc_control[0:1]     #- mpc_control_last[0:1]   # CoM forward acceleration delta
    mpc_control_acz = mpc_control[1:2]   #- mpc_control_last[1:2]   # CoM vertical acceleration delta
    mpc_control_alpha = mpc_control[2:3] #- mpc_control_last[2:3]   # angular acceleration delta
    #half_weight = config.mass * config.grav / 2.0  # nominal static load per wheel (~27 kg robot)
    mpc_control_fl = mpc_control[3:6] #/ half_weight    #- mpc_control_last[3:6]   # left contact-point force delta
    mpc_control_fr = mpc_control[6:9] #/ half_weight    #- mpc_control_last[6:9]   # right contact-point force delta

    # Computed in step() (pre-substep qpos/qvel + WBC qddot); see _joint_targets_from_qddot.
    joint_pos_des = info["joint_pos_des"]           # 6   desired leg joint positions
    wheel_vel_des = info["wheel_vel_des"]           # 2   desired wheel velocities

    q_target, dq_target = self._joint_targets_from_qddot(
        qpos_joint=joint_angles,
        qvel_joint=joint_vel,
        qddot=info["mpc_qddot"],
    )

    state = jp.hstack([
        noisy_linvel,        # 3   local linear velocity
        noisy_gyro,          # 3   body angular velocity
        noisy_gravity,       # 3   projected gravity (tilt)
        (noisy_joint_angles - self._default_pose)[self._leg_ids],         # 6   leg position error vs default
        noisy_joint_vel,     # 8   all joint velocities (incl. wheels)
        action,              # 8   previous action
        info["command"],     # 2   [forward_vel, yaw_rate]
        com_height_err,      # 1   CoM height error (com_z - target)
        #dfcip_pcom,          # 3   MPC: DFCIP state - CoM position
        #dfcip_vcom,          # 3   MPC: DFCIP state - CoM velocity
        #dfcip_c,             # 3   MPC: DFCIP state - contact-point position
        #dfcip_vc_z,          # 1   MPC: DFCIP state - contact-point vertical velocity
        #dfcip_theta,         # 1   MPC: DFCIP state - base yaw
        #dfcip_v,             # 1   MPC: DFCIP state - forward velocity
        #dfcip_omega,         # 1   MPC: DFCIP state - yaw rate
        mpc_control_a,       # 1   MPC: control - CoM forward acceleration
        mpc_control_acz,     # 1   MPC: control - CoM vertical acceleration
        mpc_control_alpha,   # 1   MPC: control - angular acceleration
        mpc_control_fl,      # 3   MPC: control - left contact force
        mpc_control_fr,      # 3   MPC: control - right contact force
        info["mpc_tau"],          # 8   MPC: WBC feedforward torque
        #joint_pos_des[self._leg_ids],       # 6   MPC: desired leg joint positions
        #wheel_vel_des[self._wheel_ids],     # 2   MPC: desired wheel velocities
    ])  

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    privileged_state = jp.hstack([
        state,                                            # 47
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
    ])  

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
        "tracking_mpc_accel": self._reward_mpc_accel(data, info),
        "tracking_mpc_alpha": self._reward_mpc_alpha(data, info),
        # Balance / body configuration.
        "orientation": self._cost_orientation(data),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "base_height": self._cost_height(body_height, info["base_height_target"]),
        "posture": self._cost_posture(data.qpos[7:], command),
        # Effort / smoothness / safety.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "dof_vel": self._cost_dof_vel(data.qvel[6:], command),
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
  
  # in _post_init / config:  mpc_tracking_sigma = 0.5   (adimensionale, come tracking_sigma)

  def _reward_mpc_accel(self, data, info):
    dt = self.dt
    lin = self.get_local_linvel(data)
    a_real = (lin[0] - info["last_local_linvel"][0]) / dt

    # MPC control: [a, ac_z, alpha, Fl(3), Fr(3)]
    a_cmd = info["mpc_control"][0]

    a_scale = 2.0  # m/s^2
    error = jp.square((a_real - a_cmd) / a_scale)

    return jp.exp(-error / self._config.reward_config.mpc_tracking_sigma)


  def _reward_mpc_alpha(self, data, info):
    dt = self.dt
    gyro = self.get_gyro(data)
    alpha_real = (gyro[2] - info["last_gyro"][2]) / dt

    # MPC control: [a, ac_z, alpha, Fl(3), Fr(3)]
    alpha_cmd = info["mpc_control"][2]

    alpha_scale = 4.0  # rad/s^2
    error = jp.square((alpha_real - alpha_cmd) / alpha_scale)

    return jp.exp(-error / self._config.reward_config.mpc_tracking_sigma)


  def _cost_orientation(self, data: mjx.Data) -> jax.Array:
    # 0 when the base is level; grows with tilt.
    return jp.sum(jp.square(self.get_gravity(data)[:2]))

  def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_height(self, body_height, base_height_target: jax.Array) -> jax.Array:
    err = body_height - base_height_target
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

  def _cost_dof_vel(self, qvel: jax.Array, command: jax.Array) -> jax.Array:
    leg_qvel = qvel[jp.array(consts.LEG_DOF_IDS)]     # 6 leg DOF, excludes wheels
    raw  = jp.sum(jp.square(leg_qvel))
    gate = jp.exp(-jp.sum(jp.square(command))
                  / 0.25 )#self._config.reward_config.dof_vel_cmd_sigma)
    return raw #* gate

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

  def sample_height(self, rng: jax.Array) -> jax.Array:
    return jax.random.uniform(
        rng,
        shape=(),
        minval=self._cmd_h[0],
        maxval=self._cmd_h[1],
    )

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