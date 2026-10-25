# Joystick task for TITA (two-wheeled balancing biped) on MuJoCo Playground / MJX.
#
# The robot has two 3-DOF legs, each ending in a driven wheel:
#   qpos[7:] index -> 0,1,2 = left  hip/thigh/knee, 3 = left  wheel
#                     4,5,6 = right hip/thigh/knee, 7 = right wheel
#   LEG_DOF_IDS = [0,1,2,4,5,6]   WHEEL_DOF_IDS = [3,7]
#
# Hybrid RL control (HRC), after Zhang et al., "Hybrid Balance Control for
# Wheeled Bipedal Robot via Residual Reinforcement Learning Optimization"
# (ACIRS 2025). The MPC/WBC/PD controls all 8 motors (tau_zeta); the policy
# adds a torque tau_pi on the 6 motors of the paper (thigh, knee, wheel per
# side; hip abduction is "not considered"):  tau_C = tau_zeta + tau_pi.
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
      episode_length=5000,  # paper: "a iteration of maximum 5000 steps" (50 s)
      randomize_reset=0.0,  # minimal setup: fixed spawn, zero initial velocity
      # Terrain. "flat": scene of the task. "slopes": paper Group 1 scene with
      # three lanes (xmls/scene_slopes.xml); each reset spawns on one of
      # `lanes`, chosen uniformly. The slope angle is never observed.
      # "mixed": xmls/scene_mixed.xml (flat, slope lanes, sparse rough boxes,
      # Perlin hfield); each reset picks a region with probability `weight`.
      terrain_config=config_dict.create(
          scene="flat",
          lane_y=[0.0, 3.0, -3.0],   # lane centres [m]: 0, 5, 10 deg
          lanes=[0, 1, 2],           # lanes used at reset (indices of lane_y)
          # Mixed regions: flat, slope 0/5/10 deg, rough, perlin.
          # centre [x, y], spawn spread (+-, m), random heading (1/0), weight.
          region_centre=[[-15.0, 0.0], [0.0, 0.0], [0.0, 3.0], [0.0, -3.0],
                         [25.0, 0.0], [45.0, 0.0]],
          region_spread=[0.5, 0.0, 0.0, 0.0, 0.5, 0.5],
          region_random_yaw=[1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
          region_weight=[0.2, 0.1, 0.1, 0.1, 0.25, 0.25],
          # Ground tilt of the scene [deg], used only to place the spawn height
          # (scene tilt10: 10). Not observed by the policy.
          tilt_deg=0.0,
      ),
      # Nominal torque tau_zeta = WBC feedforward + joint PD on the legs, as the
      # paper's basic controller (LQR on the wheels + PD pose controller on the
      # legs, kp/kd 85/12 hip, 75/15 knee). Without the PD a constant residual
      # bias of ~1-2 N m on the legs makes the robot collapse.
      #   mode "none":    tau_ff only
      #   mode "wbc":     PD on the one-sim-step WBC target (as mjx_tita.py)
      #   mode "posture": PD toward the posture reference q0 of the WBC (HRC)
      # Gains per motor: hip-abd, thigh, knee, wheel (left, then right); wheels
      # stay tau_ff only.
      nominal_pd=config_dict.create(
          mode="posture",
          Kp=[35.0, 85.0, 75.0, 0.0] * 2,
          Kd=[10.0, 12.0, 15.0, 0.0] * 2,
      ),
      action_repeat=1,
      soft_joint_pos_limit_factor=0.95,
      # Residual RL torque channel (paper: tau_C = tau_zeta + tau_pi):
      #   tau_pi    = action * tau_max   on motor_ids (6 motors, abduction excluded)
      #   tau_total = clip(tau_nominal + tau_pi, actuator_limits)
      # The paper gives no scale for tau_pi: tau_max is set from the nominal
      # torque magnitudes measured in MPC-only rollouts (see tita_hybrid_plan).
      residual_config=config_dict.create(
          enabled=True,          # False -> pure MPC/WBC (residual torque = 0)
          # Motor order: left thigh, knee, wheel, right thigh, knee, wheel.
          motor_ids=[1, 2, 3, 5, 6, 7],
          # [N m] at |action| = 1. Legs 5 N m (residual corrections of a few
          # N m suffice, cf. the MIT Residual MPC humanoid); wheels 2 N m
          # (~22 N at the ground; a constant 1 N m already drifts ~1.8 m).
          tau_max=[5.0, 5.0, 2.0, 5.0, 5.0, 2.0],
          # Optional low-pass on the applied action, a = s*a_prev + (1-s)*a_nn.
          # 0 = off (paper); raise only if the residual torque chatters.
          action_lpf=0.0,
      ),
      noise_config=config_dict.create(
          level=0.0,  # minimal setup: no sensor noise
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=0.1,
              joint_mpc_pos=0.01,
              joint_mpc_vel=0.1,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      # Paper reward (Sec. III-C): r = r_thetalim + r_bp, r_bp = r_b + r_z + r_psi.
      # Every task error goes through f(x) = 1/(e^x + e^-x + 2) - 0.25, i.e.
      # sigmoid(x)*sigmoid(-x) - 0.25 (max 0 at zero error). The paper gives no
      # weights: all task terms are unit. No desired pendulum angle theta_pd:
      # only its rate is constrained.
      reward_config=config_dict.create(
          scales=config_dict.create(
              # r_b: balance state q_b = [x_w, xdot_w, theta_p, theta_p_dot].
              balance_pos=1.0,       # x_w - x_ref
              balance_vel=1.0,       # xdot_w - v_ref
              pendulum_ang_vel=1.0,  # theta_p_dot
              # r_z: CoG height above the wheel-axis midpoint.
              cog_height=1.0,
              # r_psi: base pose q_psi = [roll, pitch, yaw], desired 0.
              roll=1.0,
              pitch=1.0,
              yaw=1.0,
              # r_thetalim: hip-forward / knee soft-limit hinge.
              dof_pos_limits=-1.0,
              # Failure. Not in the paper; the kernels are <= 0, so without it
              # falling early would end the negative return.
              termination=-500.0,
              # Not in the paper (tita_hybrid_plan_v2 fine-tune): command
              # tracking with the tita_v1 exp kernel, steep near zero error,
              # where the paper kernel f is flat. 0 = off.
              tracking_lin_vel=0.0,
              tracking_ang_vel=0.0,
          ),
          tracking_sigma=0.0625,
          only_positive_rewards=False,
          base_height_target=0.4,
      ),
      # Evaluation only (paper Fig. 7): instantaneous pushes on the base at
      # fixed times, opposite to the direction of motion (-x of the spawn
      # heading). Empty times = off.
      push_config=config_dict.create(
          times=[],          # [s], e.g. [12.0, 20.0, 32.0]
          forces=[],         # [N], e.g. [10.0, 20.0, 30.0]
          duration=0.1,      # [s] (paper: "instantaneous", not specified)
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      # Command = [forward_vel (m/s), yaw_rate (rad/s)], fed to the MPC.
      # Paper task (Group 1): move a random distance D in [1, 4] m. Here the
      # distance becomes a trapezoidal velocity profile v_ref(t) with
      # integral D (MPC is velocity-commanded); x_ref(t) is its integral.
      command_config=config_dict.create(
          # "distance": paper task (trapezoidal profile to a random distance).
          # "joystick": random [vx, wz] commands as tita_v1 (residual_tita):
          # resampled at exponential intervals and low-pass filtered (ramp).
          mode="distance",
          a=[2.5, 0.8],           # joystick: full command half-range [vx, wz]
          a_learned=[2.0, 0.6],   # joystick: inner range
          p_extend=0.2,           # joystick: probability of the extension band
          b=[0.75, 0.75],         # joystick: probability a command is non-zero
          command_lpf=0.02,       # joystick: ramp gain on the applied command
          distance=[1.0, 4.0],  # D range [m]; set both equal for a fixed target
          v_max=0.5,            # profile cruise speed [m/s] (not in the paper)
          a_max=0.5,            # profile acceleration [m/s^2] (not in the paper)
          # Evaluation only (paper Fig. 8): when True the caller writes
          # info["x_ref"] and info["command"] before each step (any trajectory);
          # the trapezoidal profile is not used.
          external_reference=False,
          h=[0.4, 0.4],         # CoM height command range [min, max] [m]
          names=["vx", "wz"]
      ),
      impl="jax",
      naconmax=4 * 8192,
      njmax=40,
  )


class Joystick(tita_base.TitaEnv):
  """Track a joystick command with the TITA wheeled biped."""

  # Order of the paper task errors e_t (see _task_errors).
  _TASK_ERROR_NAMES = (
      "x_err", "v_err", "theta_p_dot", "z_err", "roll", "pitch", "yaw")

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Overrides are applied by the base class, so read the scene from them here.
    scene = (config_overrides or {}).get("terrain_config.scene", config.terrain_config.scene)
    scene_xmls = {"slopes": "scene_slopes.xml", "mixed": "scene_mixed.xml",
                  "mixed_lite": "scene_mixed_lite.xml",
                  "mixed_noperlin": "scene_mixed_noperlin.xml",
                  "rough_sparse": "scene_rough_sparse.xml",
                  "tilt10": "scene_tilt10.xml"}
    xml_path = (consts.ROOT_PATH / "xmls" / scene_xmls[scene] if scene in scene_xmls
                else consts.task_to_xml(task))
    super().__init__(
        xml_path=xml_path.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._init_q = self._init_q.at[2].set(0.4435)
    # Leg posture reference of the nominal PD = the WBC posture task q0
    # (home keyframe); wheel entries are unused (zero gains).
    self._posture_ref = jp.array(config.q0)
    self._pd_kp = jp.array(self._config.nominal_pd.Kp)
    self._pd_kd = jp.array(self._config.nominal_pd.Kd)

    self._leg_ids = jp.array(consts.LEG_DOF_IDS)
    self._wheel_ids = jp.array(consts.WHEEL_DOF_IDS)

    # Residual motors (paper: hip forward, knee, wheel per side) and the
    # hip-forward / knee subset used by r_thetalim and the observation.
    rc = self._config.residual_config
    self._res_ids = jp.array(rc.motor_ids)
    self._res_tau_max = jp.array(rc.tau_max)
    self._res_leg_ids = jp.array([i for i in rc.motor_ids if i not in consts.WHEEL_DOF_IDS])

    # Soft joint limits (hip forward / knee; wheels are continuous).
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    f = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = self._lowers[self._res_leg_ids] * f
    self._soft_uppers = self._uppers[self._res_leg_ids] * f
    self._torque_limits = jp.array(self.mj_model.actuator_forcerange[:, 1])

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = jp.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom(consts.FLOOR_GEOM).id
    # All ground geoms (floor + ramps/plateaus of the slopes scene).
    self._ground_geom_ids = [self._floor_geom_id] + [
        i for i in range(self._mj_model.ngeom)
        if self._mj_model.geom(i).name.startswith(("slope_", "terrain_", "rough_"))]
    tc = self._config.terrain_config
    self._lane_y = jp.array([tc.lane_y[i] for i in tc.lanes])
    self._region_centre = jp.array(tc.region_centre)
    self._region_spread = jp.array(tc.region_spread)
    self._region_random_yaw = jp.array(tc.region_random_yaw)
    self._region_weight = jp.array(tc.region_weight) / sum(tc.region_weight)
    # Spawn height offset per region: ground height under the spawn centre
    # (the Perlin pad is above the floor), from a downward ray at init.
    self._region_dz = jp.zeros(len(tc.region_centre))
    if tc.scene.startswith("mixed"):
      import mujoco
      import numpy as np
      # Move the robot far away so that the rays only see the terrain.
      data0 = mujoco.MjData(self._mj_model)
      data0.qpos[0:3] = [1e3, 1e3, 10.0]
      mujoco.mj_forward(self._mj_model, data0)
      dz = []
      for cx, cy in tc.region_centre:
        geomid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(self._mj_model, data0, np.array([cx, cy, 2.0]),
                             np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        dz.append(2.0 - dist if dist >= 0 else 0.0)
      self._region_dz = jp.array(dz)
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

    self._cmd_h = jp.array(self._config.command_config.h)
    cc = self._config.command_config
    self._joystick = cc.mode == "joystick"
    self._cmd_a = jp.array(cc.a)
    self._cmd_a_learned = jp.array(cc.a_learned)
    self._cmd_p_extend = float(cc.p_extend)
    self._cmd_b = jp.array(cc.b)
    self._cmd_resample_scale = 0.5 * self._config.episode_length * self.dt

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

  def _get_noisy_joint_state(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns qpos/qvel with noise applied only to actuated joints."""
    rng, qpos_rng, qvel_rng = jax.random.split(rng, 3)

    joint_qpos_measured = (
        qpos[7:]
        + jax.random.uniform(
            qpos_rng,
            shape=qpos[7:].shape,
            minval=-1.0,
            maxval=1.0,
        )
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_mpc_pos
    )

    joint_qvel_measured = (
        qvel[6:]
        + jax.random.uniform(
            qvel_rng,
            shape=qvel[6:].shape,
            minval=-1.0,
            maxval=1.0,
        )
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_mpc_vel
    )

    qpos_measured = qpos.at[7:].set(joint_qpos_measured)
    qvel_measured = qvel.at[6:].set(joint_qvel_measured)

    return rng, qpos_measured, qvel_measured

  def _joint_targets_from_qddot(self, qpos_joint, qvel_joint, qddot):
    """NOMINAL joint targets: current joint state integrated one simulation step
    with the WBC's solved joint accelerations. Used ONLY for the nominal MPC/WBC
    outer PD -- never for the residual policy target. qddot is (1, nv) from the
    vmapped solver: [:, :6] is the floating base, [:, 6:] the nj actuated joints
    in qpos[7:]/qvel[6:] order."""
    dt = self._config.sim_dt
    qddot_joint = qddot[0, 6:]
    dq_target = qvel_joint + qddot_joint * dt
    q_target = qpos_joint + qvel_joint * dt + 0.5 * qddot_joint * dt**2
    return q_target, dq_target

  def _nominal_torque(self, q, qd, tau_ff, qddot):
    """tau_zeta: WBC feedforward + leg joint PD (see nominal_pd), all 8 motors."""
    pd = self._config.nominal_pd
    if pd.mode == "none":
      return tau_ff
    if pd.mode == "wbc":
      q_des, dq_des = self._joint_targets_from_qddot(q, qd, qddot)
    else:  # "posture": the WBC posture reference, zero joint velocity
      q_des, dq_des = self._posture_ref, jp.zeros_like(qd)
    return tau_ff + self._pd_kp * (q_des - q) + self._pd_kd * (dq_des - qd)

  def _residual_torque(self, action):
    """tau_pi: policy torque on the 6 residual motors, zero on hip abduction.
    `action` is the clipped (and optionally filtered) policy output in [-1, 1];
    action = 0 gives exactly zero residual torque, i.e. the pure MPC/WBC."""
    tau_rl = jp.zeros(self.mjx_model.nu).at[self._res_ids].set(action * self._res_tau_max)
    return tau_rl if self._config.residual_config.enabled else jp.zeros_like(tau_rl)

  def _combine_torque(self, tau_nominal, tau_rl):
    """tau_C = tau_zeta + tau_pi, clipped to the actuator limits."""
    return jp.clip(tau_nominal + tau_rl, -self._torque_limits, self._torque_limits)

  @property
  def action_size(self) -> int:
    # The policy outputs tau_pi on the residual motors only (base.py: nu).
    return len(self._config.residual_config.motor_ids)

  # --------------------------------------------------------------------
  # Task reference and balance state (paper Sec. II-III).
  # --------------------------------------------------------------------
  def _reference_profile(self, t, distance):
    """Trapezoidal (or triangular) velocity profile with integral `distance`.
    Returns (x_ref, v_ref) at time t [s]."""
    cc = self._config.command_config
    a = cc.a_max
    v_peak = jp.minimum(cc.v_max, jp.sqrt(distance * a))  # triangular if D short
    t_acc = v_peak / a
    t_cruise = (distance - v_peak * t_acc) / v_peak
    t_dec = t_acc + t_cruise
    t_end = t_dec + t_acc
    v_ref = jp.where(
        t < t_acc, a * t,
        jp.where(t < t_dec, v_peak, jp.maximum(v_peak - a * (t - t_dec), 0.0)))
    x_acc = 0.5 * a * t_acc**2
    td = jp.clip(t - t_dec, 0.0, t_acc)
    x_ref = jp.where(
        t < t_acc, 0.5 * a * t**2,
        jp.where(t < t_dec, x_acc + v_peak * (t - t_acc),
                 x_acc + v_peak * t_cruise + v_peak * td - 0.5 * a * td**2))
    x_ref = jp.where(t >= t_end, distance, x_ref)
    return x_ref, v_ref

  def _base_rpy(self, data: mjx.Data) -> jax.Array:
    """Roll, pitch, yaw of the floating base (q_psi in the paper)."""
    w, x, y, z = data.qpos[3:7]
    roll = jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jp.array([roll, pitch, yaw])

  def _balance_state(self, data: mjx.Data, info: dict):
    """Paper balance quantities in the control frame {C} (wheel-axis midpoint).

    Returns q_b = [x_w, xdot_w, theta_p, theta_p_dot], the CoG p_c / pdot_c in
    {C}, and the CoG height z_c above the wheel axis. x_w is the progress of the
    wheel midpoint along the initial heading; theta_p is the virtual-pendulum
    inclination (wheel midpoint -> CoG) from the vertical.
    """
    com = data.sensordata[self._base_com_adr]
    com_vel = data.sensordata[self._base_com_linvel_adr]
    wheels = data.geom_xpos[self._feet_geom_id]
    wheel_vel = data.sensordata[self._foot_linvel_sensor_adr]
    pivot = jp.mean(wheels, axis=0)
    pivot_vel = jp.mean(wheel_vel, axis=0)
    lateral = wheels[0] - wheels[1]
    lateral_vel = wheel_vel[0] - wheel_vel[1]
    span_sq = jp.maximum(jp.sum(jp.square(lateral[:2])), 1e-12)
    forward = jp.array([lateral[1], -lateral[0], 0.0]) / jp.sqrt(span_sq)
    left = jp.array([-forward[1], forward[0], 0.0])
    heading_rate = (lateral[0] * lateral_vel[1] - lateral[1] * lateral_vel[0]) / span_sq
    forward_dot = heading_rate * left
    left_dot = -heading_rate * forward

    # Progress along the initial heading, from the spawn wheel midpoint.
    x_w = jp.dot(pivot - info["start_pivot"], info["start_forward"])
    xdot_w = jp.dot(pivot_vel, forward)

    lean = com - pivot
    lean_vel = com_vel - pivot_vel
    lean_x = jp.dot(lean, forward)
    lean_x_dot = jp.dot(lean_vel, forward) + jp.dot(lean, forward_dot)
    theta_p = jp.arctan2(lean_x, lean[2])
    theta_p_dot = (lean[2] * lean_x_dot - lean_x * lean_vel[2]) / (
        jp.maximum(jp.square(lean_x) + jp.square(lean[2]), 1e-12))

    p_c = jp.array([lean_x, jp.dot(lean, left), lean[2]])
    pdot_c = jp.array([
        lean_x_dot,
        jp.dot(lean_vel, left) + jp.dot(lean, left_dot),
        lean_vel[2],
    ])
    q_b = jp.array([x_w, xdot_w, theta_p, theta_p_dot])
    return q_b, p_c, pdot_c

  def _task_errors(self, data: mjx.Data, info: dict) -> jax.Array:
    """e = [x_w - x_ref, xdot_w - v_ref, theta_p_dot, z_c - z_cd, roll, pitch, yaw]."""
    q_b, p_c, _ = self._balance_state(data, info)
    rpy = self._base_rpy(data)
    # The MPC height target is the CoM above the ground contact, i.e. above
    # the wheel axis plus the wheel radius.
    z_c = p_c[2] + jp.mean(self._wheel_radius)
    yaw_err = rpy[2] - info["start_yaw"]
    if self._joystick:
      # No position target; the yaw term becomes the yaw-rate tracking error.
      x_err = jp.zeros(())
      yaw_term = self.get_global_angvel(data)[2] - info["command"][1]
    else:
      x_err = q_b[0] - info["x_ref"]
      yaw_term = jp.arctan2(jp.sin(yaw_err), jp.cos(yaw_err))
    return jp.array([
        x_err,
        q_b[1] - info["command"][0],
        q_b[3],
        z_c - info["base_height_target"],
        rpy[0],
        rpy[1],
        yaw_term,
    ])

  # --------------------------------------------------------------------
  # Reset / step.
  # --------------------------------------------------------------------
  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x,y = +U(-0.5, 0.5), yaw = U(-pi, pi), both scaled by randomize_reset
    # (0 in the minimal setup: fixed spawn and heading).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5) * self._config.randomize_reset
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    yaw_scale = self._config.randomize_reset
    if self._config.terrain_config.scene == "slopes":
      # Paper Group 1: slope angle chosen at random (0/5/10 deg lane).
      rng, key = jax.random.split(rng)
      qpos = qpos.at[1].add(jax.random.choice(key, self._lane_y))
    elif self._config.terrain_config.scene.startswith("mixed"):
      # Region chosen at random; spawn around its centre on the local ground.
      rng, key, key_xy = jax.random.split(rng, 3)
      region = jax.random.choice(key, len(self._region_weight), p=self._region_weight)
      spread = self._region_spread[region]
      xy = self._region_centre[region] + jax.random.uniform(
          key_xy, (2,), minval=-spread, maxval=spread)
      qpos = qpos.at[0:2].add(xy).at[2].add(self._region_dz[region])
      yaw_scale = jp.maximum(yaw_scale, self._region_random_yaw[region])
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14) * yaw_scale
    quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))
    # Tilted ground (scene tilt10): with a cross-slope heading the two wheels
    # sit at different ground heights (+-d/2 sin(yaw) tan(tilt)); raise the
    # spawn so that the higher one touches instead of penetrating.
    tilt = self._config.terrain_config.tilt_deg
    if tilt:
      qpos = qpos.at[2].add(
          0.5 * self._wheel_base * jp.abs(jp.sin(yaw[0])) * jp.tan(jp.deg2rad(tilt)) + 0.005)

    # Start under mild planar motion. Ranges kept small for the STATIONARY task:
    # with the wheels reset to v_des=0, even vx=0.2 m/s pitches the base ~64 deg
    # in 0.5 s uncontrolled, so a large initial velocity makes every episode start
    # as a hard recovery instead of standing. Lateral (vy) self-damps via wheel
    # friction, so it is kept smaller still.
    rng, key = jax.random.split(rng)
    vx = jax.random.uniform(key, (1,), minval=-0.2, maxval=0.2) * self._config.randomize_reset
    rng, key = jax.random.split(rng)
    vy = jax.random.uniform(key, (1,), minval=-0.1, maxval=0.1) * self._config.randomize_reset
    #qvel = qvel.at[0:2].set(jp.concatenate([vx, vy]))
    rng, key = jax.random.split(rng)
    joint_vel = jax.random.uniform(key, shape=qvel[6:].shape, minval=-0.2, maxval=0.2,) * self._config.randomize_reset
    qvel = qvel.at[6:].set(joint_vel)

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

    # Paper task: random target distance D; the MPC gets v_ref(t) of a
    # trapezoidal profile with integral D, starting from rest.
    rng, key = jax.random.split(rng)
    target_distance = self.sample_distance(key)
    x_ref, v_ref = self._reference_profile(0.0, target_distance)
    cmd = jp.array([v_ref, 0.0])

    # Joystick task (tita_v1): random target command, applied through a ramp
    # that starts from zero; resampled at exponential intervals.
    rng, key1, key2 = jax.random.split(rng, 3)
    steps_until_next_cmd = jp.round(
        jax.random.exponential(key1) * self._cmd_resample_scale / self.dt).astype(jp.int32)
    target_command = jax.random.uniform(
        key2, shape=(2,), minval=-self._cmd_a, maxval=self._cmd_a)
    if self._joystick:
      cmd = jp.zeros(2)

    rng, height_rng = jax.random.split(rng)
    base_height_target = self.sample_height(height_rng)

    # ------------------------------------------------------------------
    # MPC augmentation: initialise the persistent MPC state and run one
    # passive pass so all MPC info fields are populated with correct shapes.
    # The command fed to the MPC starts at zero (matches info["command"]).
    # ------------------------------------------------------------------

    _, qpos_measured, qvel_measured = (
        self._get_noisy_joint_state(
            data.qpos,
            data.qvel,
            rng,
        )
    )
    tita_state0 = self.build_tita_state(data)
    dfcip_state0, theta0 = self.get_dfip_current_state(
        tita_state0,
        jp.array(0.0),
    )

    # Initialize the complete MPC warm-start from the actual reset state.
    mpc_state = self.mpc.init_state(dfcip_state0[None, :])
    mpc_state, tita_state, dfcip_state, mpc_tau, mpc_qddot, mpc_fl, mpc_fr, desired, theta_prev, mpc_reference, _solver_bad = self._run_mpc_wbc(
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

    # tau_zeta at the reset state: the policy observes the nominal torque the
    # MPC/WBC is about to apply (paper s_tau contains tau_zeta^t).
    tau_nominal = self._nominal_torque(
        qpos_measured[7:], qvel_measured[6:], mpc_tau, mpc_qddot)

    # Spawn frame of the task: wheel-axis midpoint, heading and yaw.
    wheels = data.geom_xpos[self._feet_geom_id]
    lateral = wheels[0] - wheels[1]
    start_forward = jp.array([lateral[1], -lateral[0], 0.0]) / jp.linalg.norm(lateral[:2])

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "base_height_target": base_height_target,
        # Paper task: target distance, position reference and spawn frame.
        "target_distance": target_distance,
        "target_command": target_command,
        "steps_until_next_cmd": steps_until_next_cmd,
        "x_ref": x_ref,
        "start_pivot": jp.mean(wheels, axis=0),
        "start_forward": start_forward,
        "start_yaw": self._base_rpy(data)[2],
        "task_errors": jp.zeros(7),       # e_t
        "task_errors_last": jp.zeros(7),  # e_{t-1}
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
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
            "qpos_measured": qpos_measured,
            "qvel_measured": qvel_measured,
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
        # Torques (paper s_tau): tau_zeta^t at the current state, tau_zeta^{t-1},
        # and the tau_pi / tau_C applied during the last control step.
        "tau_nominal": tau_nominal,
        "tau_nominal_last": tau_nominal,
        "tau_residual": jp.zeros(self.mjx_model.nu),
        "tau_total": tau_nominal,
        "tau_saturated_frac": jp.zeros(()),
        "mpc_bad": jp.zeros(()),                      # 1.0 if the MPC/WBC solve failed this step
        "mpc_fallback_count": jp.zeros((), dtype=jp.int32),  # cumulative solver failures
        "nonfinite_count": jp.zeros((), dtype=jp.int32),     # cumulative non-finite physics states
        "reward_terms" : {}
    }
    info["task_errors"] = self._task_errors(data, info)
    info["task_errors_last"] = info["task_errors"]

    dummy_rewards = self._get_reward(
        data,
        jp.zeros(self.action_size),
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
    for k in self._TASK_ERROR_NAMES:
      metrics[f"task/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info, jp.zeros(self.action_size))
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

    # Detect a genuine solver failure BEFORE masking with nan_to_num (Section 17).
    # Only the APPLIED outputs are checked: tau/qddot (applied torque), x0 (obs)
    # and the reference (obs). The mpc_state warm-start is NOT checked -- its
    # D0_shifted (FDDP shooting defects) is non-finite by design at unused nodes,
    # so checking the whole pytree would fire the fallback every step and freeze
    # the controller (the standalone driver threads the same state and never
    # checks it).
    solver_bad = (
        jp.any(~jp.isfinite(tau))
        | jp.any(~jp.isfinite(qddot))
        | jp.any(~jp.isfinite(x0))
        | jp.any(~jp.isfinite(reference))
    )
    tau = jp.nan_to_num(tau[0], nan=0.0, posinf=0.0, neginf=0.0)
    qddot = jp.nan_to_num(qddot, nan=0.0, posinf=0.0, neginf=0.0)
    return mpc_state, tita_state, x0, tau, qddot, fl, fr, desired, theta_prev, reference, solver_bad

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    if self._config.push_config.times:
      state = self._apply_scheduled_push(state)

    action = jp.clip(action, -1.0, 1.0)
    # Optional low-pass on the applied action (s = 0: off, as in the paper).
    s = self._config.residual_config.action_lpf
    action = s * state.info["last_act"] + (1.0 - s) * action
    tau_rl = self._residual_torque(action)

    # The MPC/WBC solution for this control step was computed at the end of
    # the previous step (or at reset) on the current state, so the policy has
    # already observed the nominal torque tau_zeta^t it is correcting.
    tau = state.info["mpc_tau"]
    qddot = state.info["mpc_qddot"]

    # PRNG stream for the 500 Hz substep loop (consumed by the outer-PD noise).
    rng = state.info["rng"]

    def substep_fn(data, _):
        data, rng = data

        rng, q_measured, qd_measured = (
            self._get_noisy_joint_state(
                data.qpos,
                data.qvel,
                rng,
            )
        )

        q = q_measured[7:]
        qd = qd_measured[6:]
        # Refresh the nominal outer-PD target every substep (500 Hz) from the
        # current joint state and the held WBC qddot, matching the standalone.
        tau_nom = self._nominal_torque(q, qd, tau, qddot)
        ctrl = self._combine_torque(tau_nom, tau_rl)
        data = data.replace(ctrl=ctrl)
        data = mjx.step(self.mjx_model, data)
        return (data, rng), ctrl

    (data, rng), tau_tot_s = jax.lax.scan(
        substep_fn, (state.data, rng), xs=None, length=self.n_substeps)
    state = state.replace(data=data)

    # Torque diagnostics from the last substep.
    tau_tot_d = tau_tot_s[-1]
    saturated = jp.abs(tau_tot_d) >= (self._torque_limits - 1e-3)
    state.info["tau_residual"] = tau_rl
    state.info["tau_total"] = tau_tot_d
    state.info["tau_saturated_frac"] = jp.mean(
        saturated.astype(state.info["tau_saturated_frac"].dtype))

    # Advance the task reference to the new time: v_ref is the MPC command of
    # the next control step, x_ref its integral. With an external reference
    # the caller has already written both for the next step.
    if self._joystick:
      # Resample the target at exponential intervals; ramp the applied command.
      state.info["steps_until_next_cmd"] -= 1
      state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
      resample = state.info["steps_until_next_cmd"] <= 0
      state.info["target_command"] = jp.where(
          resample, self.sample_command(key1, state.info["target_command"]),
          state.info["target_command"])
      state.info["steps_until_next_cmd"] = jp.where(
          resample,
          jp.round(jax.random.exponential(key2) * self._cmd_resample_scale / self.dt).astype(jp.int32),
          state.info["steps_until_next_cmd"])
      lpf = self._config.command_config.command_lpf
      state.info["command"] = (
          state.info["command"]
          + lpf * (state.info["target_command"] - state.info["command"]))
    elif not self._config.command_config.external_reference:
      x_ref, v_ref = self._reference_profile(
          (state.info["step"] + 1) * self.dt, state.info["target_distance"])
      state.info["x_ref"] = x_ref
      state.info["command"] = jp.array([v_ref, 0.0])

    # Solve the MPC/WBC on the new state for the next control step.
    rng, qpos_measured, qvel_measured = (
        self._get_noisy_joint_state(
            data.qpos,
            data.qvel,
            rng,
        )
    )
    state.info["rng"] = rng

    new_mpc_state, tita_state, dfcip_state, new_tau, new_qddot, mpc_fl, mpc_fr, desired, theta_prev, mpc_reference, solver_bad = self._run_mpc_wbc(
        data=data,
        qpos=qpos_measured,
        qvel=qvel_measured,
        command=state.info["command"],
        base_height_target=state.info["base_height_target"],
        action=action,
        mpc_state=state.info["mpc_state"],
        theta_prev=state.info["theta_prev"],
        timestep=state.info["step"] + 1
    )

    bad = solver_bad
    state.info["mpc_bad"] = bad.astype(state.info["mpc_bad"].dtype)
    state.info["mpc_fallback_count"] = (
        state.info["mpc_fallback_count"]
        + bad.astype(state.info["mpc_fallback_count"].dtype))
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

    # tau_zeta^t at the new state, observed by the policy before it acts.
    state.info["tau_nominal_last"] = state.info["tau_nominal"]
    state.info["tau_nominal"] = self._nominal_torque(
        qpos_measured[7:], qvel_measured[6:], tau, qddot)

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
        "qpos_measured": qpos_measured,
        "qvel_measured": qvel_measured,
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

    # Paper motion error e_t (and e_{t-1}) against the advanced reference.
    state.info["task_errors_last"] = state.info["task_errors"]
    state.info["task_errors"] = self._task_errors(data, state.info)

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

    # NaN-robustness (Section 17): a diverged physics state (hard fall / blow-up)
    # must terminate the episode with a finite penalty and must NEVER leak NaN
    # into the observation or the reward, or it poisons the PPO update. Detect
    # BEFORE sanitising with nan_to_num, and count it.
    finite = (
        jp.all(jp.isfinite(data.qpos))
        & jp.all(jp.isfinite(data.qvel))
        & jp.all(jp.isfinite(obs["state"]))
        & jp.all(jp.isfinite(obs["privileged_state"]))
        & jp.isfinite(reward)
    )
    done = jp.logical_or(done, ~finite)
    # Penalty commensurate with a normal termination (the termination term is
    # -100 * dt = -1.0 in the summed reward). A large raw -100 here would dwarf
    # the ~0.01/step normal reward and blow up the value target variance -> NaN.
    reward = jp.where(finite, reward, -1.0)
    obs = jax.tree_util.tree_map(
        lambda x: jp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), obs)
    state.info["nonfinite_count"] = (
        state.info["nonfinite_count"]
        + (~finite).astype(state.info["nonfinite_count"].dtype))

    state.info["reward_terms"] = rewards

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    for k, v in zip(self._TASK_ERROR_NAMES, state.info["task_errors"]):
      state.metrics[f"task/{k}"] = v

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

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Isaac: contatto sui body di terminazione (base). In più: caduta.
    fall_termination = self.get_upvector(data)[-1] < 0.0
    base_contact = jp.array([
        geoms_colliding(data, gid, ground_id)
        for gid in self._termination_geom_id
        for ground_id in self._ground_geom_ids
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
    info["q_measured"] = noisy_joint_angles

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )
    info["qd_measured"] = noisy_joint_vel

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
    # Normalize the contact forces by the nominal static load per wheel
    # (m*g/2 ~= 136 N) so they enter the observation at ~O(1) instead of ~136,
    # keeping the observation well scaled.
    half_weight = config.mass * config.grav / 2.0  # ~135.8 N per wheel
    mpc_control_fl = mpc_control[3:6] / half_weight    # left contact-point force / (m g / 2)
    mpc_control_fr = mpc_control[6:9] / half_weight    # right contact-point force / (m g / 2)

    # Paper state s_t = [s_r, s_g, s_e, s_tau] (Sec. III-A). The desired
    # inclination theta_pd is left out on purpose, as in the paper.
    q_b, p_c, pdot_c = self._balance_state(data, info)
    x_ref, yaw_des = info["x_ref"], info["start_yaw"]
    if self._joystick:
      # No position or heading target in the joystick task: x_w (unbounded),
      # x_ref and the spawn heading are not observed.
      q_b = q_b.at[0].set(0.0)
      x_ref, yaw_des = jp.zeros(()), jp.zeros(())
    ids = self._res_ids
    tau_scale = self._torque_limits[ids]

    state = jp.hstack([
        # s_r: robot state
        q_b,                 # 4   [x_w, xdot_w, theta_p, theta_p_dot]
        p_c,                 # 3   CoG position in {C}
        pdot_c,              # 3   CoG velocity in {C}
        noisy_joint_angles[self._res_leg_ids],  # 4   hip-forward / knee angles
        noisy_joint_vel[ids],                   # 6   hip-forward / knee / wheel velocities
        # s_g: desired balance state and pose (no theta_pd)
        x_ref,               # 1   x_wd
        info["command"][0],  # 1   xdot_wd = v_ref
        info["base_height_target"],  # 1   z_cd
        0.0,                 # 1   zdot_cd
        jp.zeros(2),         # 2   roll_d, pitch_d
        yaw_des,             # 1   yaw_d
        # s_e: motion error and history
        info["task_errors"],       # 7   e_t
        info["task_errors_last"],  # 7   e_{t-1}
        # s_tau: torque and history, normalized by the actuator limits
        info["tau_nominal"][ids] / tau_scale,       # 6   tau_zeta^t
        info["tau_nominal_last"][ids] / tau_scale,  # 6   tau_zeta^{t-1}
        info["tau_residual"][ids] / tau_scale,      # 6   tau_pi^{t-1}
        info["tau_total"][ids] / tau_scale,         # 6   tau_C^{t-1}
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    privileged_state = jp.hstack([
        state,                                            # 65
        gyro,                                             # 3
        accelerometer,                                    # 3
        gravity,                                          # 3
        linvel,                                           # 3
        angvel,                                           # 3
        joint_angles[self._leg_ids],                      # 6
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
    del action, first_contact, contact
    # Paper r_bp: each component of e_t through the kernel f.
    f = self._task_error_kernel(info["task_errors"])
    return {
        # r_b: balance.
        "balance_pos": f[0],
        "balance_vel": f[1],
        "pendulum_ang_vel": f[2],
        # r_z: CoG height.
        "cog_height": f[3],
        # r_psi: base pose.
        "roll": f[4],
        "pitch": f[5],
        "yaw": f[6],
        # r_thetalim + failure.
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "termination": self._cost_termination(done),
        # Command tracking (exp kernel, see reward_config.scales).
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)),
    }

  @staticmethod
  def _task_error_kernel(error: jax.Array) -> jax.Array:
    """Paper f(x) = 1/(e^x + e^-x + 2) - 0.25 = sigmoid(x)*sigmoid(-x) - 0.25.
    Bell-shaped, maximum 0 at zero error, -0.25 for large errors."""
    return jax.nn.sigmoid(error) * jax.nn.sigmoid(-error) - 0.25

  def _cost_residual(self, action: jax.Array) -> jax.Array:
    # Weak residual-magnitude penalty (paper-aligned). Discourages chattering /
    # gratuitously large residuals without forcing the action to zero (which,
    # under the default-pose target, would be the zero-action overshoot).
    return jp.sum(jp.square(action))

  def _reward_tracking_lin_vel(
      self, command: jax.Array, local_vel: jax.Array
  ) -> jax.Array:
    # Command-normalized error (Residual-MPC paper): dividing by (1+|cmd|) keeps
    # the Gaussian kernel from saturating to ~0 at large commands, so the policy
    # still gets a tracking gradient in the extension region (vx > 1.5).
    err = jp.square((command[0] - local_vel[0]) / (1.0 + jp.abs(command[0])))
    return jp.exp(-err / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, command: jax.Array, gyro: jax.Array
  ) -> jax.Array:
    err = jp.square((command[1] - gyro[2]) / (1.0 + jp.abs(command[1])))
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

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    # Dimensionless action difference (no 1/dt): the global x dt then leaves a
    # clean per-step magnitude instead of inverting to 1/dt.
    return jp.sum(jp.square(act - last_act))

  def _cost_action_rate_2nd(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    """Second difference of the action, i.e. discrete jerk."""
    return jp.sum(jp.square(act - 2.0 * last_act + last_last_act))


  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Paper r_thetalim: hip-forward and knee joints only.
    q = qpos[self._res_leg_ids]
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
    """tita_v1 command sampler: mostly in [-a_learned, a_learned], with
    probability p_extend in the extension band [a_learned, a] (either sign);
    each component kept / zeroed / resampled at random."""
    rng, in_rng, band_rng, sign_rng, ext_rng, w_rng, z_rng = jax.random.split(rng, 7)
    n = self._cmd_a.shape[0]
    inner = jax.random.uniform(
        in_rng, (n,), minval=-self._cmd_a_learned, maxval=self._cmd_a_learned)
    band = jax.random.uniform(
        band_rng, (n,), minval=self._cmd_a_learned, maxval=self._cmd_a)
    sign = jp.where(jax.random.bernoulli(sign_rng, 0.5, (n,)), 1.0, -1.0)
    is_ext = jax.random.bernoulli(ext_rng, self._cmd_p_extend, (n,))
    y_k = jp.where(is_ext, sign * band, inner)
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(n,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(n,))
    return x_k - w_k * (x_k - y_k * z_k)

  def sample_distance(self, rng: jax.Array) -> jax.Array:
    # Paper Group 1: random movement target in [1, 4] m.
    d_min, d_max = self._config.command_config.distance
    return jax.random.uniform(rng, shape=(), minval=d_min, maxval=d_max)

  def sample_height(self, rng: jax.Array) -> jax.Array:
    return jax.random.uniform(
        rng,
        shape=(),
        minval=self._cmd_h[0],
        maxval=self._cmd_h[1],
    )

  def _apply_scheduled_push(self, state: mjx_env.State) -> mjx_env.State:
    """Paper Fig. 7: push of forces[i] N during [times[i], times[i] + duration),
    on the base, opposite to the spawn heading (the direction of motion)."""
    pc = self._config.push_config
    t = state.info["step"] * self.dt
    starts = jp.array(pc.times)
    active = (t >= starts) & (t < starts + pc.duration)
    magnitude = jp.sum(jp.where(active, jp.array(pc.forces), 0.0))
    xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
    xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
        -magnitude * state.info["start_forward"])
    return state.replace(data=state.data.replace(xfrc_applied=xfrc_applied))

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