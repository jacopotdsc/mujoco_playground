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
import numpy as np
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


def _normalize_action_deadband(value, num_actuators: int) -> jax.Array:
  """Accept legacy scalars or one normalized threshold per motor (also in 6D)."""
  values = np.asarray(value, dtype=float)
  if values.ndim == 0:
    values = np.full(num_actuators, values.item())
  if values.shape != (num_actuators,):
    raise ValueError(f"action_deadband must be scalar or shape {(num_actuators,)}, got {values.shape}")
  if not np.all(np.isfinite(values)) or np.any((values < 0) | (values >= 1)):
    raise ValueError("action_deadband entries must be finite and in [0, 1)")
  return jp.asarray(values)


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
      randomize_reset=0.0,
      # Terrains to use. One entry loads that terrain's own scene; the mixed
      # ["flat", "rough"] setup loads scene_all.xml (flat floor plus the sparse
      # rough field, without Perlin) and each reset chooses one spawn centre
      # uniformly from the listed terrains.
      terrain=["flat", "rough"],
      # Region centre [x, y] of each terrain inside scene_all.xml; must match
      # the shifts the XML was built with. All three spawn pads are at z=0,
      # so the spawn height stays the keyframe one.
      terrain_spawn_points=config_dict.create(
          flat=[0.0, 0.0],
          rough=[20.0, 0.0],
      ),
      # Experimental boxes placed in the flat spawn region when flat is among
      # the selected terrains (including mixed flat/rough training).
      sparse_obstacles=config_dict.create(
          enabled=True, mode="multi", height=0.015, terrain_xml=""),
      # MPC/WBC targets and policy offsets use the same PD gains from
      # residual_config, so the controller tracks one complete target:
      # q_des_wbc + action * action_scale_pos (and likewise for velocity).
      action_repeat=1,
      leg_only_actions=False,  # True: policy controls six leg joints; wheels stay nominal.
      # Residual RL scales: leg position offset (rad) and wheel velocity target
      # (rad/s), applied to the clipped policy action about the accepted WBC targets.
      # Arrays follow motor order: left hip/thigh/knee/wheel, right hip/thigh/knee/wheel.
      # Position scaling uses leg entries. Wheel actions offset the accepted
      # WBC wheel-speed target by at most 5 rad/s (= 0.4625 m/s at r=0.0925 m).
      action_scale_pos=[0.20, 0.35, 0.50, 0.0] * 2,
      action_scale_vel=[0.0, 0.0, 0.0, 5.0] * 2,
      # Motor order: [hip, thigh, knee, wheel] left, then right.
      # Ignore small residual outputs; keep the selected threshold unchanged
      # while tuning physical correction strength from torque and balance data.
      action_deadband=[0.05, 0.05, 0.05, 0.05] * 2,
      soft_joint_pos_limit_factor=0.95,
      # Residual RL torque channel, added on top of the nominal torque:
      #   tau_total = clip(tau_nominal + tau_residual, actuator_limits)
      # The policy amplitude is set only by action_scale_pos/action_scale_vel.
      residual_config=config_dict.create(
          enabled=True,          # False -> pure MPC/WBC (residual torque = 0)
          Kp=[20.0, 20.0, 20.0, 0.0] * 2,              # shared leg proportional gain
          Kd=[1.0, 1.0, 1.0, 0.0] * 2,                 # shared leg derivative gain
          # At full policy action: 2 Nm/(rad/s) * 5 rad/s = 10 Nm.
          Kd_wheel=[0.0, 0.0, 0.0, 2.0] * 2,           # shared wheel velocity gain
      ),
      noise_config=config_dict.create(
          level=0.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=0.1,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      # Active reward terms and weights ported from Lite3.
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking. The policy's whole job is to beat the MPC here.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Task-space balance/posture rewards (kernel already negative).
              wheel_vel_tracking=1.0,  # Forward wheel-axis velocity error.
              pendulum_ang_vel=2.0,  # Virtual pendulum angular velocity.
              cog_vel_z=0.0,  # Vertical CoG velocity error.
              # Posture.
              orientation=-0.0,
              base_height=-0.0,
              lin_vel_z=-0.0,
              ang_vel_x=-0.00,
              ang_vel_y=-0.00,
              stance_width=0.0,
              dof_pos_limits=-0.0,
              termination=-500.0,
              residual_torque=-0.000,
              action_rate=-0.00,
              action_rate_2nd=-0.000,
              torques=0.0,
              energy=0.0,
          ),
          tracking_sigma=0.25,
          stance_width=config.d,
          stance_width_sigma=0.10,
          balance_posture=config_dict.create(
              # Error order: [vx_w - command_vx, theta_p_dot], [vz_c - target_vz].
              # Unit weights/scales are explicit defaults, not paper coefficients.
              balance_weights=[1.0, 1.0],
              vertical_velocity_weight=1.0,
              vertical_velocity_target=0.0,
          ),
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      # Command = [forward_vel (m/s), yaw_rate (rad/s)].
      # vx range reaches into the extension region (the nominal MPC/WBC tracks
      # to ~1.5 m/s and falls at >=2.0), so the residual policy is exposed to
      # commands only it can satisfy. yaw range covers the full nominal envelope.
      command_config=config_dict.create(
          fixed_target=None,  # Optional [vx, wz]; command still starts at zero and follows the LPF.
          # RAMP protocol (train == eval == deploy): the applied command
          # low-pass-tracks the target with gain command_lpf. Under a ramp the
          # nominal MPC/WBC reaches ~3.0-3.5 m/s (vs ~1.5 under a step), so the
          # command range is set toward that ramp limit: the residual learns
          # where the nominal actually struggles (~2.5-3.5).
          command_lpf=0.02,       # 0.02 = ramp (train/eval/deploy); 1.0 = instant step
          a=[2.5, 0.8],           # full command half-range: vx +-2.0, wz +-0.8 (B's proven range)
          a_learned=[2.0, 0.6],   # survivable inner range
          p_extend=0.2,           # ~70% of commands in [-a_learned, a_learned], ~30% in
                                  # the extension band [a_learned, a] (either sign), so
                                  # most episodes are survivable (strong, stable signal)
                                  # while the residual still practices the hard region.
          b=[0.75, 0.75],    # prob a resampled command stays non-zero
          h=[0.4, 0.4],    # CoM height command range [min, max] [m]
          names=["vx", "wz"]
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
    terrains = list((config_overrides or {}).get("terrain", config.terrain))
    # Legacy registry entries (TitaJoystickRoughTerrain, ...) select the
    # terrain through `task`; they apply while config.terrain is the default.
    if terrains == ["flat"] and task != "flat_terrain":
      terrains = [task.removesuffix("_terrain")]
    if (not terrains or len(set(terrains)) != len(terrains)
        or not set(terrains) <= set(consts.TERRAIN_XMLS)):
      raise ValueError(
          f"terrain must be a non-empty list of distinct names from "
          f"{list(consts.TERRAIN_XMLS)}, got {terrains!r}")
    if len(terrains) > 1 and set(terrains) != {"flat", "rough"}:
      raise ValueError(
          "Mixed Tita training supports only ['flat', 'rough']; "
          "Perlin is available only as a single-terrain environment.")
    xml_path = (consts.TERRAIN_XMLS[terrains[0]] if len(terrains) == 1
                else consts.ALL_TERRAIN_XML)
    super().__init__(
        xml_path=xml_path.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._terrains = terrains
    if "flat" in terrains and self._config.sparse_obstacles.enabled:
      self._add_sparse_obstacles()
    self._post_init()

  def _add_sparse_obstacles(self) -> None:
    """Add the three experimental flat-terrain boxes to the in-memory model."""
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import mujoco

    root = ET.fromstring(Path(self._xml_path).read_text())
    world = root.find("worldbody")
    mode = self._config.sparse_obstacles.mode
    configured_height = float(self._config.sparse_obstacles.height)
    if configured_height <= 0:
      raise ValueError("sparse_obstacles.height must be positive")
    if mode == "terrain_overlay":
      terrain_xml = self._config.sparse_obstacles.terrain_xml
      if not terrain_xml:
        raise ValueError("terrain_overlay requires sparse_obstacles.terrain_xml")
      source_root = ET.parse(terrain_xml).getroot()
      source_world = source_root.find("worldbody")
      if source_world is None:
        raise ValueError(f"No worldbody in terrain XML: {terrain_xml}")
      copied = 0
      for geom in source_world.findall("geom"):
        if geom.get("type", "sphere") == "plane":
          continue  # retain the flat scene's supporting plane
        attrs = dict(geom.attrib)
        attrs["name"] = f"rough_overlay_{copied}"
        ET.SubElement(world, "geom", attrs)
        copied += 1
      if copied == 0:
        raise ValueError(f"No terrain geoms found in {terrain_xml}")
      print(f"[Tita terrain overlay] copied {copied} geoms from {terrain_xml}")
      obstacles = ()
    elif mode == "multi":
      # x, y, height, half-width [m]; legacy experimental layout.
      obstacles = (
          (1.5, 0.0, 0.015, 0.30),
          (-1.5, 1.0, 0.020, 0.30),
          (0.0, -2.0, 0.025, 0.30),
      )
    elif mode == "single_left":
      # Fixed 1.5 cm step under the left wheel lane only.  With
      # randomize_reset=0 the robot starts at x=y=0 with fixed heading.
      obstacles = ((1.5, self._config.reward_config.stance_width / 2,
                    configured_height, 0.10),)
    elif mode == "double":
      # Same step across both wheel lanes.
      obstacles = ((1.5, 0.0, configured_height, 0.60),)
    else:
      raise ValueError(
          "sparse_obstacles.mode must be 'multi', 'single_left', 'double', or 'terrain_overlay', "
          f"got {mode!r}")
    for index, (x, y, height, half_width) in enumerate(obstacles):
      ET.SubElement(world, "geom", name=f"residual_training_obstacle_{index}",
                    type="box", pos=f"{x} {y} {height / 2}",
                    size=f"0.15 {half_width} {height / 2}",
                    contype="1", conaffinity="0", priority="1",
                    friction="0.6", condim="3", rgba="0.5 0.4 0.3 1")
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=tita_base.get_assets())
    model.opt.timestep = self._config.sim_dt
    model.vis.global_.offwidth = self._mj_model.vis.global_.offwidth
    model.vis.global_.offheight = self._mj_model.vis.global_.offheight
    self._mj_model = model
    self._mjx_model = mjx.put_model(model)
    self._imu_site_id = model.site("imu").id

  def _post_init(self) -> None:
    self._action_deadband = _normalize_action_deadband(
        self._config.action_deadband, self.mjx_model.nu)
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._init_q = self._init_q.at[2].set(0.4435)
    self._spawn_points = jp.array(
        [self._config.terrain_spawn_points[t] for t in self._terrains],
        dtype=self._init_q.dtype)
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
    # Ground geoms present in the scene and, per wheel, its contact sensors
    # on each of them (scene_all.xml has one ground per region).
    model_geoms = {self._mj_model.geom(i).name for i in range(self._mj_model.ngeom)}
    floors = [g for g in consts.FLOOR_GEOMS if g in model_geoms]
    self._floor_geom_ids = [self._mj_model.geom(g).id for g in floors]
    self._feet_floor_found_sensor = [
        [self._mj_model.sensor(f"{foot}_{g}_found").id for g in floors]
        for foot in ("FL", "FR")
    ]
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
    self._cmd_a_learned = jp.array(self._config.command_config.a_learned)
    self._cmd_p_extend = float(self._config.command_config.p_extend)
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

  def _get_noisy_joint_state(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns qpos/qvel with noise applied only to actuated joints."""

    # --- Actuated joints -----------------------------------------------------
    joint_angles = qpos[7:]
    rng, noise_rng = jax.random.split(rng)
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = qvel[6:]
    rng, noise_rng = jax.random.split(rng)
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    qpos_measured = qpos.at[7:].set(noisy_joint_angles)
    qvel_measured = qvel.at[6:].set(noisy_joint_vel)

    return rng, qpos_measured, qvel_measured

  def _joint_targets_from_qddot(self, qpos_joint, qvel_joint, qddot):
    """Integrate the accepted WBC acceleration for one simulation tick."""
    dt = self._config.sim_dt
    qddot_joint = qddot[0, 6:]
    dq_target = qvel_joint + qddot_joint * dt
    q_target = qpos_joint + qvel_joint * dt + 0.5 * qddot_joint * dt**2
    return q_target, dq_target

  def _residual_joint_targets(self, action, q_des_wbc, dq_des_wbc):
    """Policy offsets around the accepted (or held) WBC reference."""

    q_des_offset = action[self._leg_ids] * jp.broadcast_to(jp.asarray(self._config.action_scale_pos), action.shape)[self._leg_ids]
    dq_des_offset = action[self._wheel_ids] * jp.broadcast_to(jp.asarray(self._config.action_scale_vel), action.shape)[self._wheel_ids]
    q_des_rl = q_des_wbc.at[self._leg_ids].add(q_des_offset)
    dq_des_rl = dq_des_wbc.at[self._wheel_ids].add(dq_des_offset)
    return q_des_rl, dq_des_rl

  def _combine_torque(
      self, q, qd, tau_ff, q_des_wbc, dq_des_wbc, q_des_rl, dq_des_rl
  ):
    """Add an action-zero residual to the complete nominal WBC controller."""
    rc = self._config.residual_config
    kp = jp.asarray(rc.Kp)
    kd = jp.asarray(rc.Kd) + jp.asarray(rc.Kd_wheel)

    # Split the single complete-target PD law only to expose tau_residual in
    # diagnostics. Algebraically, tau_nominal + tau_residual is the PD torque
    # for q_des_rl and dq_des_rl.
    tau_nominal = (
        tau_ff
        + kp * (q_des_wbc - q)
        + kd * (dq_des_wbc - qd)
    )

    # The learned channel contains only the torque generated by policy offsets.
    # Therefore q_des_rl=q_des_wbc and dq_des_rl=dq_des_wbc imply tau_rl=0.
    tau_res = (
        kp * (q_des_rl - q_des_wbc)
        + kd * (dq_des_rl - dq_des_wbc)
    )
    if self._config.leg_only_actions:
      tau_res = tau_res.at[self._wheel_ids].set(0.0)
    tau_rl = tau_res if rc.enabled else jp.zeros_like(tau_res)

    tau_total = jp.clip(tau_nominal + tau_rl, -self._torque_limits, self._torque_limits)
    return tau_nominal, tau_rl, tau_total

  # --------------------------------------------------------------------
  # Reset / step.
  # --------------------------------------------------------------------
  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x,y = +U(-0.5, 0.5), yaw = U(-pi, pi).
    rng, key = jax.random.split(rng)
    dxy = (
        jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        * self._config.randomize_reset
    )
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    if len(self._terrains) > 1:
      # Spawn at the centre of one listed terrain of scene_all.xml, uniformly.
      rng, key = jax.random.split(rng)
      spawn = jax.random.choice(key, self._spawn_points)
      qpos = qpos.at[0:2].add(spawn)
    rng, key = jax.random.split(rng)
    yaw = jp.zeros((1,))  # Fixed initial heading.
    quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
    #qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

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

    if self._config.command_config.fixed_target is not None:
      cmd = jp.asarray(self._config.command_config.fixed_target, dtype=cmd.dtype)

    rng, height_rng = jax.random.split(rng)
    base_height_target = self.sample_height(height_rng)

    # ------------------------------------------------------------------
    # MPC augmentation: initialise the persistent MPC state and run one
    # passive pass so all MPC info fields are populated with correct shapes.
    # The command fed to the MPC starts at zero (matches info["command"]).
    # ------------------------------------------------------------------

    rng, qpos_measured, qvel_measured = (
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
    initial_mpc_state = mpc_state
    mpc_state, tita_state, dfcip_state, mpc_tau, mpc_qddot, mpc_fl, mpc_fr, desired, theta_prev, mpc_reference, _solver_bad = self._run_mpc_wbc(
        data=data,
        qpos=qpos_measured,
        qvel=qvel_measured,
        command=jp.zeros_like(cmd),
        base_height_target=base_height_target,
        mpc_state=mpc_state,
        theta_prev=0.0,
        timestep=0
        )

    # No valid predecessor at reset: zero feedforward and measured position /
    # velocity references are the fallback until the first successful solve.
    mpc_state = jax.tree_util.tree_map(
        lambda new, old: jp.where(_solver_bad, old, new), mpc_state, initial_mpc_state)
    mpc_tau = jp.where(_solver_bad, jp.zeros_like(mpc_tau), mpc_tau)
    mpc_qddot = jp.where(_solver_bad, jp.zeros_like(mpc_qddot), mpc_qddot)
    mpc_fl = jp.where(_solver_bad, jp.zeros_like(mpc_fl), mpc_fl)
    mpc_fr = jp.where(_solver_bad, jp.zeros_like(mpc_fr), mpc_fr)
    desired = jp.where(_solver_bad, jp.zeros_like(desired), desired)
    mpc_reference = jp.where(_solver_bad, jp.zeros_like(mpc_reference), mpc_reference)
    theta_prev = jp.where(_solver_bad, theta0, theta_prev)
    q_des_wbc, dq_des_wbc = self._joint_targets_from_qddot(
        qpos_measured[7:], qvel_measured[6:], mpc_qddot)
    q_des_wbc = jp.where(_solver_bad, qpos_measured[7:], q_des_wbc)
    dq_des_wbc = jp.where(_solver_bad, qvel_measured[6:], dq_des_wbc)
    q_des, dq_des = self._residual_joint_targets(jp.zeros(self.mjx_model.nu), q_des_wbc, dq_des_wbc)

    info = {
        "rng": rng,
        "qpos_measured": qpos_measured,
        "qvel_measured": qvel_measured,
        "step": 0,
        "command": jp.zeros_like(cmd),
        "base_height_target": base_height_target,
        "target_command" : cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "policy_action": jp.zeros(self.mjx_model.nu),
        "applied_action": jp.zeros(self.mjx_model.nu),
        "balance_errors": jp.zeros(2),
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
        "q_des_wbc": q_des_wbc,
        "dq_des_wbc": dq_des_wbc,
        "joint_pos_des": q_des,
        "wheel_vel_des": dq_des,
        # Per-step torque diagnostics (logged only; set every step()).
        "tau_nominal": jp.zeros(self.mjx_model.nu),
        "tau_residual": jp.zeros(self.mjx_model.nu),
        "tau_ff_last": jp.zeros(self.mjx_model.nu),  # Previous control step, raw Nm.
        "tau_rl_last": jp.zeros(self.mjx_model.nu),  # Previous control step, raw Nm.
        "tau_ff_last_last": jp.zeros(self.mjx_model.nu),  # Two control steps ago, raw Nm.
        "tau_rl_last_last": jp.zeros(self.mjx_model.nu),  # Two control steps ago, raw Nm.
        "tau_total": jp.zeros(self.mjx_model.nu),
        "tau_saturated_frac": jp.zeros(()),
        "mpc_bad": _solver_bad.astype(data.qpos.dtype),                      # 1.0 if the MPC/WBC solve failed this step
        "mpc_fallback_count": _solver_bad.astype(jp.int32),  # cumulative solver failures
        "nonfinite_count": jp.zeros((), dtype=jp.int32),     # cumulative non-finite physics states
        "reward_terms" : {}
    }

    balance_errors, _ = self._balance_posture_errors(data, info)
    info["balance_errors"] = balance_errors

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
    info["r_bp"] = sum(dummy_rewards[k] for k in ("wheel_vel_tracking", "pendulum_ang_vel", "cog_vel_z"))

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    metrics["reward/r_bp"] = jp.zeros(())
    metrics["balance/wheel_vel_error"] = info["balance_errors"][0]
    metrics["balance/pendulum_ang_vel"] = info["balance_errors"][1]
    obs = self._get_obs(data, info, jp.zeros(self.mjx_model.nu))
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def _run_mpc_wbc(self, data, qpos: jax.Array, qvel: jax.Array, command: jax.Array, base_height_target, mpc_state, theta_prev, timestep: int):
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

    mpc_state, tau, qddot, fl, fr, desired = self.mpc.whole_body_run(
        mpc_state,
        x0,
        jp.asarray(qpos)[None, :],
        jp.asarray(qvel)[None, :],
        pl_world_wbc,
        pr_world_wbc,
        dpl_world_wbc,
        dpr_world_wbc,
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

  @property
  def action_size(self) -> int:
    return len(consts.LEG_DOF_IDS) if self._config.leg_only_actions else self.mjx_model.nu

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if action.shape != (self.action_size,):
      raise ValueError(f"Expected action shape {(self.action_size,)}, got {action.shape}")
    if self._config.leg_only_actions:
      action = jp.zeros(self.mjx_model.nu, dtype=action.dtype).at[self._leg_ids].set(action)
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)

    policy_action = jp.clip(action, -1.0, 1.0)
    state.info["policy_action"] = policy_action
    action = jp.where(
        jp.abs(policy_action) < self._action_deadband,
        jp.zeros_like(policy_action),
        policy_action,
    )
    state.info["applied_action"] = action

    # Shared encoder sample: the policy and first PD substep see this too.
    qpos_measured = state.info["qpos_measured"]
    qvel_measured = state.info["qvel_measured"]

    # PRNG stream for the 500 Hz substep loop (consumed by the outer-PD noise).
    rng = state.info["rng"]

    new_mpc_state, tita_state, dfcip_state, new_tau, new_qddot, mpc_fl, mpc_fr, desired, theta_prev, mpc_reference, solver_bad = self._run_mpc_wbc(
        data=state.data,
        qpos=qpos_measured,
        qvel=qvel_measured,
        command=state.info["command"],
        base_height_target=state.info["base_height_target"],
        mpc_state=state.info["mpc_state"],
        theta_prev=state.info["theta_prev"],
        timestep=state.info["step"]
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
    state.info["theta_prev"]    = jp.where(bad, state.info["theta_prev"], theta_prev)
    state.info["tita_state"]    = tita_state
    state.info["dfcip_state"]   = dfcip_state
    state.info["mpc_grf_left"]  = jp.where(bad, state.info["mpc_grf_left"], mpc_fl)
    state.info["mpc_grf_right"] = jp.where(bad, state.info["mpc_grf_right"], mpc_fr)
    state.info["mpc_control"]   = jp.where(bad, state.info["mpc_control"], mpc_reference[0][0, 13:])
    state.info["mpc_desired"]   = jp.where(bad, state.info["mpc_desired"], desired[0])

    # Select once per MPC update; hold both references throughout the substeps.
    new_q_des_wbc, new_dq_des_wbc = self._joint_targets_from_qddot(qpos_measured[7:], qvel_measured[6:], new_qddot)
    state.info["q_des_wbc"] = jp.where(bad, state.info["q_des_wbc"], new_q_des_wbc)
    state.info["dq_des_wbc"] = jp.where(bad, state.info["dq_des_wbc"], new_dq_des_wbc)
    q_des_wbc = state.info["q_des_wbc"]
    dq_des_wbc = state.info["dq_des_wbc"]
    q_des_rl, dq_des_rl = self._residual_joint_targets(action, q_des_wbc, dq_des_wbc)

    def substep_fn(carry, _):
        data, rng, qpos_measured, qvel_measured = carry
        q = qpos_measured[7:]
        qd = qvel_measured[6:]
        tau_nom, tau_rl, ctrl = self._combine_torque(
            q, qd, tau, q_des_wbc, dq_des_wbc, q_des_rl, dq_des_rl
        )
        data = data.replace(ctrl=ctrl)
        data = mjx.step(self.mjx_model, data)
        
        rng, qpos_measured, qvel_measured = self._get_noisy_joint_state(data.qpos, data.qvel, rng)
        return (data, rng, qpos_measured, qvel_measured), (tau_nom, tau_rl, ctrl)

    (data, rng, qpos_measured, qvel_measured), (tau_nom_s, tau_rl_s, tau_tot_s) = jax.lax.scan(
        substep_fn, (state.data, rng, qpos_measured, qvel_measured),
        xs=None, length=self.n_substeps)
    state = state.replace(data=data)
    state.info["rng"] = rng
    state.info["qpos_measured"] = qpos_measured
    state.info["qvel_measured"] = qvel_measured

    # Torque diagnostics from the last substep (logged only; no separate subgraph).
    tau_nom_d, tau_rl_d, tau_tot_d = tau_nom_s[-1], tau_rl_s[-1], tau_tot_s[-1]
    saturated = jp.abs(tau_tot_d) >= (self._torque_limits - 1e-3)
    # Shift before writing the current torques, so obs can access both steps.
    state.info["tau_ff_last_last"] = state.info["tau_ff_last"]
    state.info["tau_rl_last_last"] = state.info["tau_rl_last"]
    state.info["tau_ff_last"] = state.info["tau_nominal"]
    state.info["tau_rl_last"] = state.info["tau_residual"]
    state.info["tau_nominal"] = tau_nom_d
    state.info["tau_residual"] = tau_rl_d
    state.info["tau_total"] = tau_tot_d
    state.info["tau_saturated_frac"] = jp.mean(
        saturated.astype(state.info["tau_saturated_frac"].dtype))

    state.info["joint_pos_des"] = q_des_rl
    state.info["wheel_vel_des"] = dq_des_rl

    # Foot contact bookkeeping (kept for eval/plotting).
    contact = jp.array([
        jp.any(jp.array([data.sensordata[self._mj_model.sensor_adr[sid]] > 0
                         for sid in foot_sensors]))
        for foot_sensors in self._feet_floor_found_sensor
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

    # Expose the same continuous task errors used by r_b to the policy.
    balance_errors, _ = self._balance_posture_errors(data, state.info)
    state.info["balance_errors"] = balance_errors

    obs = self._get_obs(data, state.info, action)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    # The requested task kernel is negative even at zero error. Preserve its
    # learning signal instead of flattening all negative totals to zero.
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
    # Diagnostic aggregate only: do not add r_bp again to the summed reward.
    state.info["r_bp"] = sum(rewards[k] for k in ("wheel_vel_tracking", "pendulum_ang_vel", "cog_vel_z"))
    state.metrics["reward/r_bp"] = state.info["r_bp"]

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["balance/wheel_vel_error"] = state.info["balance_errors"][0]
    state.metrics["balance/pendulum_ang_vel"] = state.info["balance_errors"][1]

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
    # NOTE: was "-= 0" (in-progress/debug edit), which disabled the
    # countdown entirely -- commands would only ever resample once the
    # counter happened to already be <=0 from initialization, never again
    # after that.
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["target_command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["target_command"]),
        state.info["target_command"],
    )
    
    lpf = self._config.command_config.command_lpf
    state.info["command"] = (
        state.info["command"]
        + lpf * (state.info["target_command"] - state.info["command"])
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
        geoms_colliding(data, gid, floor_id)
        for gid in self._termination_geom_id
        for floor_id in self._floor_geom_ids
    ]).any()
    return fall_termination | base_contact

  # --------------------------------------------------------------------
  # Observations (proprio "state" + "privileged_state").
  # --------------------------------------------------------------------

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], action: jax.Array
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
    joint_vel = data.qvel[6:]
    noisy_joint_angles = info["qpos_measured"][7:]
    noisy_joint_vel = info["qvel_measured"][6:]

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

    # Last applied control: complete nominal baseline and learned residual.
    tau_ff = info["tau_nominal"] / self._torque_limits
    tau_rl = info["tau_residual"] / self._torque_limits

    state = jp.hstack([
        # robot state
        noisy_linvel,        # 3   local linear velocity
        noisy_gyro,          # 3   body angular velocity
        noisy_gravity,       # 3   projected gravity (tilt)
        noisy_joint_angles[self._leg_ids],         # 6   measured leg joint angles
        noisy_joint_vel,     # 8   all joint velocities (incl. wheels)
        # action history: previous action, not torque history
        action,              # 8   previous action
        #info["last_act"],    # 8   previous action
        # desired balance: commanded motion (current observation subset)
        info["command"],     # 2   [forward_vel, yaw_rate]
        # motion error: current height error; no error history yet
        com_height_err,      # 1   CoM height error (com_z - target)
        info["balance_errors"],  # 2   [wheel velocity error, pendulum ang. velocity]
        info["q_des_wbc"][self._leg_ids],  # 6   WBC desired leg positions (wheel angle omitted: unbounded)
        info["dq_des_wbc"],               # 8   WBC desired velocities (legs + wheels)
        #dfcip_pcom,          # 3   MPC: DFCIP state - CoM position
        #dfcip_vcom,          # 3   MPC: DFCIP state - CoM velocity
        #dfcip_c,             # 3   MPC: DFCIP state - contact-point position
        #dfcip_vc_z,          # 1   MPC: DFCIP state - contact-point vertical velocity
        #dfcip_theta,         # 1   MPC: DFCIP state - base yaw
        #dfcip_v,             # 1   MPC: DFCIP state - forward velocity
        #dfcip_omega,         # 1   MPC: DFCIP state - yaw rate
        # additional MPC control inputs
        # mpc_control_a,       # 1   MPC: control - CoM forward acceleration
        # mpc_control_acz,     # 1   MPC: control - CoM vertical acceleration
        # mpc_control_alpha,   # 1   MPC: control - angular acceleration
        # mpc_control_fl,      # 3   MPC: control - left contact force
        # mpc_control_fr,      # 3   MPC: control - right contact force
        # torque feedback from the last substep (zero at reset)
        #tau_ff,             # 8   nominal WBC+PD torque / actuator maximum
        #tau_rl,             # 8   residual torque / per-actuator maximum torque
        # info["tau_ff_last"] / self._torque_limits,  # 8   previous nominal torque
        # info["tau_rl_last"] / self._torque_limits,  # 8   previous residual torque
        # info["tau_ff_last_last"] / self._torque_limits,  # 8   nominal torque two steps ago
        # info["tau_rl_last_last"] / self._torque_limits,  # 8   residual torque two steps ago
        #joint_pos_des[self._leg_ids],       # 6   MPC: desired leg joint positions
        #wheel_vel_des[self._wheel_ids],     # 2   MPC: desired wheel velocities
    ])  

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    privileged_state = jp.hstack([
        state,                                            # 50
        gyro,                                             # 3
        accelerometer,                                    # 3
        gravity,                                          # 3
        linvel,                                           # 3
        angvel,                                           # 3
        joint_angles[self._leg_ids],  # 6
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

    return {
        **self._reward_balance_posture(data, info),
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "base_height": self._cost_height(
            data.sensordata[self._base_com_adr][2], info["base_height_target"]
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_x": self._cost_ang_vel_x(self.get_global_angvel(data)),
        "ang_vel_y": self._cost_ang_vel_y(self.get_global_angvel(data)),
        "stance_width": self._reward_stance_width(
            data.geom_xpos[self._feet_geom_id]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "termination": self._cost_termination(done),
        "residual_torque": self._cost_residual_torque(info["tau_residual"]),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "action_rate_2nd": self._cost_action_rate_2nd(
            action, info["last_act"], info["last_last_act"]
        ),
        "torques": self._cost_torques(data.actuator_force),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
    }
  @staticmethod
  def _task_error_kernel(error: jax.Array, offset: float = 0.25) -> jax.Array:
    """Stable bell kernel; offset=0.25 gives a maximum of zero."""
    return jax.nn.sigmoid(error) * jax.nn.sigmoid(-error) - offset

  def _balance_posture_errors(self, data: mjx.Data, info: dict):
    """Task errors from the current DFCIP geometry, not stale MPC diagnostics.

    Pendulum pivot = midpoint of the wheel axes. Its angle is distinct from
    base pitch and from DFCIP theta (heading). There is no x-position error
    and no desired pendulum angle.
    """
    com = data.sensordata[self._base_com_adr]
    com_vel = data.sensordata[self._base_com_linvel_adr]
    wheels = data.geom_xpos[self._feet_geom_id]
    wheel_vel = data.sensordata[self._foot_linvel_sensor_adr]
    pivot = jp.mean(wheels, axis=0)
    pivot_vel = jp.mean(wheel_vel, axis=0)
    lateral = wheels[0] - wheels[1]
    lateral_vel = wheel_vel[0] - wheel_vel[1]
    # Squared horizontal wheel separation [m^2]; guard against division by zero.
    wheel_span_xy_sq = jp.maximum(jp.sum(jp.square(lateral[:2])), 1e-12)
    forward = jp.array([lateral[1], -lateral[0], 0.0]) / jp.sqrt(wheel_span_xy_sq)
    heading_rate = (
        lateral[0] * lateral_vel[1] - lateral[1] * lateral_vel[0]
    ) / wheel_span_xy_sq
    # Heading rotation only corrects the moving sagittal frame; no yaw penalty.
    forward_dot = heading_rate * jp.array([-forward[1], forward[0], 0.0])
    lean = com - pivot
    lean_vel = com_vel - pivot_vel
    lean_x = jp.dot(lean, forward)
    lean_x_dot = jp.dot(lean_vel, forward) + jp.dot(lean, forward_dot)
    # Analytic derivative of atan2(lean_x, lean_z), including heading motion.
    theta_p_dot = (lean[2] * lean_x_dot - lean_x * lean_vel[2]) / (
        jp.maximum(jp.square(lean_x) + jp.square(lean[2]), 1e-12)
    )
    cfg = self._config.reward_config.balance_posture
    e_b = jp.array([jp.dot(pivot_vel, forward) - info["command"][0], theta_p_dot])
    # CoG height is already penalized by base_height.
    e_z = jp.array([com_vel[2] - cfg.vertical_velocity_target])
    return e_b, e_z

  def _reward_balance_posture(self, data: mjx.Data, info: dict) -> dict:
    e_b, e_z = self._balance_posture_errors(data, info)
    cfg = self._config.reward_config.balance_posture
    return {
        "wheel_vel_tracking": cfg.balance_weights[0] * self._task_error_kernel(e_b[0]),
        "pendulum_ang_vel": cfg.balance_weights[1] * self._task_error_kernel(e_b[1]),
        "cog_vel_z": cfg.vertical_velocity_weight * jp.sum(self._task_error_kernel(e_z, offset=0.25)),
    }

  def _cost_residual_torque(self, tau_rl: jax.Array) -> jax.Array:
    """Keep the residual small: the MPC baseline must stay a good solution."""
    return jp.mean(jp.square(tau_rl))

  def _cost_action_rate_2nd(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    """Second difference of the action, i.e. discrete jerk."""
    return jp.sum(jp.square(act - 2.0 * last_act + last_last_act))
  # Tracking rewards.
  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    target_xy = jp.array([commands[0], 0.0])
    lin_vel_error = jp.sum(jp.square(target_xy - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)
  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[1] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)
  # Base-related rewards.
  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])
  def _cost_ang_vel_x(self, global_angvel) -> jax.Array:
    # Base angular velocity about world x, independently weighted.
    return jp.square(global_angvel[0])

  def _cost_ang_vel_y(self, global_angvel) -> jax.Array:
    # Base angular velocity about world y, independently weighted.
    return jp.square(global_angvel[1])
  def _cost_height(self, body_height, base_height_target: jax.Array) -> jax.Array:
    err = body_height - base_height_target
    return 1.0 - jp.exp(-jp.square(err / 0.05))

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

  def _reward_stance_width(self, wheel_positions: jax.Array) -> jax.Array:
    """Reward wheel-center separation near the MPC's nominal width.

    World-space distance is invariant to robot translation and heading and
    leaves the individual leg joint angles free to adapt.
    """
    width = jp.linalg.norm(wheel_positions[0] - wheel_positions[1])
    error = (width - self._config.reward_config.stance_width) / (
        self._config.reward_config.stance_width_sigma
    )
    return jp.exp(-jp.square(error))

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    q = qpos[self._leg_ids]
    out = -jp.clip(q - self._soft_lowers, None, 0.0)
    out += jp.clip(q - self._soft_uppers, 0.0, None)
    return jp.sum(out)

  # --------------------------------------------------------------------
  # Commands and perturbations.
  # --------------------------------------------------------------------

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    if self._config.command_config.fixed_target is not None:
      return jp.asarray(self._config.command_config.fixed_target, dtype=x_k.dtype)
    rng, in_rng, band_rng, sign_rng, ext_rng, w_rng, z_rng = jax.random.split(rng, 7)
    cmd_shape = self._cmd_a.shape[0]
    # Stratified magnitude: majority in the survivable inner range, a minority in
    # the extension band [a_learned, a] with random sign.
    inner = jax.random.uniform(
        in_rng, (cmd_shape,), minval=-self._cmd_a_learned, maxval=self._cmd_a_learned)
    band = jax.random.uniform(
        band_rng, (cmd_shape,), minval=self._cmd_a_learned, maxval=self._cmd_a)
    sign = jp.where(jax.random.bernoulli(sign_rng, 0.5, (cmd_shape,)), 1.0, -1.0)
    is_ext = jax.random.bernoulli(ext_rng, self._cmd_p_extend, (cmd_shape,))
    y_k = jp.where(is_ext, sign * band, inner)
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
