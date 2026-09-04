# Audit: Tita E2E vs. Residual (MPC+RL) Pipeline

Read-only technical audit. All claims are cited `file:line`. Two repos:
- `mujoco_playground` (branch `aliengo`) — env code, root `/home/mattia/Desktop/repo_jacopo/mujoco_playground`
- `mpx` (branch `tita`) — training entry point, root `/home/mattia/Desktop/repo_jacopo/mpx`

**Scope correction (important):** `train_srbd.py` does **not** import `rl_env_srbd.py`, `mjx_policy_tita.py`, `tita.py`/`mjx_tita.py` (examples), `config_srbd.py`, or `mpc_wrapper_srbd.py`. Its default/used env is `"TitaJoystickFlatTerrain"`, loaded via `mujoco_playground.registry.load(...)` (`mpx/mpx/examples/train_srbd.py:185`), which resolves to `mujoco_playground/_src/locomotion/tita/joystick.py` (`mujoco_playground/_src/locomotion/__init__.py:34,84`). That file imports `mpx.config.config_dfcip` and `mpx.utils.mpc_wrapper_dfcip.BatchedMPCControllerWrapper` (`joystick.py:21-24`) — the **DFCIP** variants, not the SRBD ones named in the task brief. `rl_env_srbd.py`/`mjx_policy_tita.py`/`config_srbd.py`/`mpc_wrapper_srbd.py` appear to be a separate/unused example pipeline for this repo state. This report analyzes the pipeline that is actually reachable from `train_srbd.py`.

Also, contrary to the brief's assumption, `joystickE2E.py` does **not** inherit from `joystick.py`. Both are **independent siblings**, each defining a class `Joystick` that subclasses `tita_base.TitaEnv` directly:
- `mujoco_playground/_src/locomotion/tita/joystickE2E.py:102` — pure end-to-end policy (no external planner).
- `mujoco_playground/_src/locomotion/tita/joystick.py:119` — MPC/WBC + residual-RL variant.
- `mujoco_playground/_src/locomotion/tita/base.py:40` — shared `TitaEnv(mjx_env.MjxEnv)` base (XML loading, sensor accessors).
- `mujoco_playground/_src/mjx_env.py:218` — root `MjxEnv` base (`dt`, `n_substeps`, `observation_size`).

Registry (`mujoco_playground/_src/locomotion/__init__.py:83-96,141-145,175-179`): `TitaJoystickE2EFlatTerrain → joystickE2E.Joystick`; `TitaJoystickFlatTerrain`/`RoughTerrain`/`StairsTerrain`/`PerlinTerrain → joystick.Joystick`. Both use the same `tita/randomize.py` domain-randomization function.

---

## 1. E2E pipeline (`joystickE2E.py`)

**Observation — actor `"state"`, 34-dim** (`joystickE2E.py:451-540`, concat at `:509-518`):

| component | dim | notes |
|---|---|---|
| `noisy_linvel` | 3 | local linear velocity, sensor `local_linvel` |
| `noisy_gyro` | 3 | body angular velocity |
| `noisy_gravity` | 3 | projected gravity (tilt), `base.py:78-79` |
| `leg_pos_err` | 6 | `(noisy_joint_angles - default_pose)[leg_ids]`, leg joints only |
| `noisy_joint_vel` | 8 | all 8 DOF incl. wheels |
| `action` | 8 | previous action, raw |
| `info["command"]` | 2 | `[forward_vel, yaw_rate]`, smoothed |
| `com_height_err` | 1 | CoM height − target |

Noise model: `noisy_x = x + (2U(0,1)-1) * noise_config.level * scale_x` (`joystickE2E.py:456-499`), scales `joint_pos=0.01, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1` (`:57-62`). `noise_config.level` defaults to **0.0** (`:55`) → observation noise is **disabled by default**. No obs history/stacking.

**Critic `"privileged_state"`** (`joystickE2E.py:523-538`): `state`(34) + clean `gyro`(3), `accelerometer`(3), clean `gravity`(3), clean `linvel`(3), global `angvel`(3), clean `leg_pos_err`(6), clean `joint_vel`(8), `actuator_force`(8), `last_contact`(2), `feet_vel`(6), `feet_air_time`(2), `current_com_height`(1), `xfrc_applied`(3) → **85-dim total** (an inline comment claims "33"/"84" — stale, the actual `hstack` is 85; `joystickE2E.py:524`).

**Action space:** dim 8 = `mjx_model.nu` (`base.py:124-125`). Not raw torque, not a direct joint-pos target — split by DOF group:
- Leg DOFs (`LEG_DOF_IDS=(0,1,2,4,5,6)`, `tita_constants.py:81`): `q_des = default_pose + action*action_scale_pos`.
- Wheel DOFs (`WHEEL_DOF_IDS=(3,7)`, `:82`): `v_des = action*action_scale_vel` (`joystickE2E.py:320-321`).
- `action_scale_pos=0.5`, `action_scale_vel=25.0` (`joystickE2E.py:47,52`). No explicit `jp.clip` on raw action in `step()`.

**Action → torque:** PD/velocity law, recomputed every physics substep (not held constant) via `jax.lax.scan` (`joystickE2E.py:323-338`):
```
tau_leg[leg]   = Kp*(q_des-q) - Kd*qd         # Kp=50.0, Kd=1.0
tau_wheel[wheel] = Kd_wheel*(v_des-qd)        # Kd_wheel=0.5
```
(`joystickE2E.py:43-45`). Gains come from `default_config()`, **not** the XML (actuators are plain `<motor gear="1">`, `tita.xml:12,16,20,24`). `ctrl_dt=0.01` (100 Hz control), `sim_dt=0.002` (500 Hz physics) → `n_substeps=5` (`joystickE2E.py:40-41`; `mjx_env.py:262-274`). XML: `Euler` integrator, `iterations=10, ls_iterations=5` (`tita.xml:2`). Hard bound: `ctrlrange`/`forcerange = [-120,120]` N·m per actuator (`tita.xml:12,16,20,24`), enforced by MuJoCo. A `self._torque_limits` array is precomputed but **never used again** (`joystickE2E.py:135`) — dead code.

**Reward** (`joystickE2E.py:545-574`), each term × weight, summed, × `dt`, clipped to `[-10000,10000]` (`:393-400`):

| term | formula | weight | cite |
|---|---|---|---|
| `tracking_lin_vel` | `exp(-(cmd_x-v_x)^2/0.0625)` | **+1.0** | `:576-580`, `:66` |
| `tracking_ang_vel` | `exp(-(cmd_yaw-ω_z)^2/0.0625)` | **+0.5** | `:582-586`, `:67` |
| `orientation` | `sum(gravity_xy^2)` | **-1.0** | `:588-590`, `:68` |
| `ang_vel_xy` | `sum(global_ω_xy^2)` | **-0.3** | `:592-593`, `:69` |
| `base_height` | `1-exp(-((h-h_tgt)/0.05)^2)` | **-1.0** | `:595-597`, `:70` |
| `posture` | `sum(w_i*(q_leg-default)^2)*exp(-‖cmd‖²/0.25)` (command-gated), `w=[1,.5,.5,1,.5,.5]` | **-1.0** | `:599-610`, `:71` |
| `torques` | `sum(actuator_force^2)` | **-1e-4** | `:612-613`, `:72` |
| `action_rate` | `sum((a-a_prev)^2)` | **-0.01** | `:615-618`, `:73` |
| `dof_pos_limits` | soft-limit hinge, `soft_factor=0.95` | **-1.0** | `:620-624`, `:53/74` |
| `dof_vel` | `sum(leg_qvel^2)` — gate computed but not applied (multiplication commented out) | **-0.0 (disabled)** | `:626-631`, `:75` |
| `termination` | `done` | **-100.0** | `:633-634`, `:76` |

**Command sampling** (`joystickE2E.py:90-95,640-649`): `command=[vx, yaw_rate]`. `a=[1.0,0.5]` → vx∈[-1,1] m/s, yaw∈[-0.5,0.5] rad/s (no vy — differential-drive biped). `b=[0.75,0.75]` (25% chance forced to 0 on resample). Height sample `h=[0.4,0.4]` (degenerate, always 0.4 m). `p_stand=0.2` field exists but is **never referenced** (`:94`) — dead config. Resample process: `x_{k+1}=x_k - w_k(x_k-y_k z_k)`, Bernoulli `w_k~0.5`; wait time `Exponential(1)*5.0`s (`:237`). Commanded value fed to policy is low-pass filtered: `command += 0.02*(target-command)` per step (`:424-427`). **No curriculum** (fixed ranges throughout training).

**Termination** (`joystickE2E.py:438-445`): `upvector_z < 0` OR `base_link_collision` geom touching floor. No tilt-angle threshold, no torque/joint-limit termination. Episode timeout `episode_length=1000` steps (10 s) is enforced by the external Brax wrapper, not in `step()`.

**Domain randomization** (`mujoco_playground/_src/locomotion/tita/randomize.py:24-102`, wired to E2E too): floor friction `U(0.4,1.0)`; DOF frictionloss `×U(0.9,1.1)`; DOF armature `×U(1.0,1.05)`; body mass `×U(0.9,1.1)`; torso mass `+U(-1,1)` kg; `qpos0[7:] += U(-0.05,0.05)` rad. Push perturbations gated by `pert_config.enable`, **default `False`** (`joystickE2E.py:84`). Obs noise (§ above) also default-off.

**PPO wiring:** `mujoco_playground/config/locomotion_params.py` has **no `elif` branch for `"TitaJoystickE2EFlatTerrain"`** (verified by grep), so its native `brax_ppo_config` falls through to the generic default (`policy=(128,128,128,128)`, `value=(256,256,256,256,256)`, `value_obs_key="state"` — i.e. **not** asymmetric there). **However**, checkpoint evidence (§7) shows the actual E2E checkpoints were trained via `train_srbd.py` itself (env name swapped to the E2E registry key), using `train_srbd.py`'s own hardcoded `PPO_PARAMS` (§6) — so `locomotion_params.py`'s branch is very likely dead code for the runs that actually exist on disk, not the operative config.

---

## 2. `train_srbd.py` residual pipeline — literal call graph

Per control step, `Joystick.step(state, action)` (`mujoco_playground/_src/locomotion/tita/joystick.py:517`):

1. Optional perturbation (`pert_config.enable`, default `False`).
2. **WBC/MPC solve — independent of the RL action**: `_run_mpc_wbc(...)` (`joystick.py:477-515`) calls `self.mpc.run(...)` (MPC trajectory opt, `:488`) then `self.mpc.whole_body_run(..., use_nn=False)` (`:500-511`) — note the NN-action argument is literally commented out (`:509`: `#scaled_action[None, :],`) and `use_nn=False` is hardcoded. Inside `mpc_wrapper_dfcip.py`, the NN action is only injected into the WBC target when `use_nn=True` (gated at `mpc_wrapper_dfcip.py:270-284,350-353`), which never happens here. **Result: `tau` (the WBC feedforward torque) has zero dependence on the RL action.**
3. NaN/solver-failure guard: on non-finite MPC output, falls back to previous step's `tau`/`qddot`/`mpc_state` (`joystick.py:533-544`).
4. `state.info["mpc_tau"] = tau` is stored (`:552`) — this is both fed into the observation (§3) and added to the applied torque (step 6).
5. **Action → joint targets** (`joystick.py:568-573`, `_compute_joint_desired`, `:273-282`):
   ```python
   q_target = self._default_pose            # fixed home pose, NOT the MPC's own planned trajectory
   dq_target = jp.zeros_like(action)
   q_des  = q_target  + action * action_scale_pos   # action_scale_pos = 0.5
   dq_des = dq_target + action * action_scale_vel   # action_scale_vel = 25.0
   ```
   (`joystick.py:66-67,273-282`).
6. **Torque combination — the exact load-bearing lines** (`joystick.py:586-593`, inside `substep_fn`, run 5× per control step via `jax.lax.scan`):
   ```python
   tau_leg   = self._config.Kp * (q_des - q) - self._config.Kd * qd     # Kp=50.0, Kd=1.0
   tau_wheel = self._config.Kd_wheel * (dq_des - qd)                    # Kd_wheel=0.5
   ctrl_nn = jp.zeros(self.mjx_model.nu)
   ctrl_nn = ctrl_nn.at[self._leg_ids].set(tau_leg[self._leg_ids])
   ctrl_nn = ctrl_nn.at[self._wheel_ids].set(tau_wheel[self._wheel_ids])
   ctrl = ctrl_nn + tau                          # <<< torque combination
   data = data.replace(ctrl=ctrl)
   data = mjx.step(self.mjx_model, data)
   ```
   Verified directly by reading `joystick.py:470-599`.
7. Observation (`_get_obs`, `:717-857`), termination (`:704-711`), reward (`:862-983`), command resampling (`:684-698`).
8. Wrapped by `mujoco_playground._src.wrapper.wrap_for_brax_training` (`train_srbd.py:183`, used at `:1338`), then standard Brax PPO rollout/update (`ppo.train`, `train_srbd.py:1351`). Brax package internals (GAE, clip-surrogate loss) were not locally inspectable (not installed in the audit sandbox) — taken as standard Brax semantics for unspecified hyperparameters, not re-verified from source.

**Is it literally `tau_final = tau_WBC + tau_NN`?** Structurally yes — it is an unconditional, ungated element-wise sum `ctrl = ctrl_nn + tau` with no blending coefficient, no gate, no "replace" branch reachable in `step()` (a `use_only_mpc` info key referenced elsewhere in `train_srbd.py:794-802,861-869` is never read inside `joystick.py`'s `step()` — dead/no-op for this env). **But** `ctrl_nn` is **not** the raw or scaled network output — it is a **PD torque tracking an action-derived target built around the fixed default pose** (`q_target = self._default_pose`, `joystick.py:274`), not around the MPC's own planned/optimal joint trajectory. And `tau_WBC` (`tau`) is itself computed with the RL action's influence path (`use_nn`) explicitly disabled. So the two terms summed are: (a) a WBC torque wholly blind to the RL action, and (b) a PD-tracked offset-from-default-pose driven by the RL action — not "MPC plan + correction on top of that same plan."

---

## 3. Actor / critic observations for the residual policy

**Actor `"state"`, 51-dim** (`joystick.py:811-835`):

`noisy_linvel`(3) + `noisy_gyro`(3) + `noisy_gravity`(3) + `leg_pos_err`(6, noisy) + `noisy_joint_vel`(8) + `action`(8, **previous** action) + `command`(2) + `com_height_err`(1) + `mpc_control_a`(1, MPC CoM fwd-accel plan) + `mpc_control_acz`(1) + `mpc_control_alpha`(1) + `mpc_control_fl`(3, left contact-force plan) + `mpc_control_fr`(3, right contact-force plan) + `info["mpc_tau"]`(8, **WBC feedforward torque, i.e. tau_WBC itself**).

Several `dfcip_pcom/vcom/c/vc_z/theta/v/omega` delta terms and `joint_pos_des`/`wheel_vel_des` are computed but **commented out of the actual concat** (`joystick.py:820-826,833-834`) — not in the vector despite being computed.

→ **`tau_WBC`/MPC output IS in the actor observation** (`mpc_tau`, `mpc_control_a/acz/alpha/fl/fr`). **Previous action (previous residual) IS included** (`action`, `:817`). **No explicit WBC task-error term** and no `joint_pos_des`/`wheel_vel_des` (both commented out).

**Critic `"privileged_state"`, 102-dim** (`joystick.py:840-855`): `state`(51, though an inline comment mislabels it "47" — stale, `:841`) + clean `gyro`(3) + `accelerometer`(3) + clean `gravity`(3) + clean `linvel`(3) + global `angvel`(3) + clean `leg_pos_err`(6) + clean `joint_vel`(8) + `actuator_force`(8, **measured/realized total torque**, a proxy for `tau_final` as applied, not `tau_WBC`/`tau_NN` individually) + `last_contact`(2) + `feet_vel`(6) + `feet_air_time`(2) + `current_com_height`(1) + `xfrc_applied`(3, push force).

**PPO wiring confirms this split is real**, not just available: `network_factory` in `train_srbd.py:139-143` sets `policy_obs_key="state"`, `value_obs_key="privileged_state"` — genuinely asymmetric actor/critic.

---

## 4. Residual reward terms and weights

`_get_reward` (`joystick.py:862-983`), weights from `default_config().reward_config.scales` (`joystick.py:79-100`). Sum × `dt=0.01` (`:664`), clipped `[-10000,10000]`, `only_positive_rewards=False` (`:95`):

| term | weight | formula | cite |
|---|---|---|---|
| `tracking_lin_vel` | **1.0** | `exp(-(cmd_x - v_x)^2 / 0.0625)` | `:895-899`, weight `:81` |
| `tracking_ang_vel` | **0.5** | `exp(-(cmd_yaw - ω_z)^2 / 0.0625)` | `:901-905`, `:82` |
| `tracking_mpc_accel` | **0.0 (disabled)** | `exp(-((a_real-a_cmd)/2.0)^2/0.0625)`, `a_real=(v_x-v_x_prev)/dt`, `a_cmd=info["mpc_control"][0]` | `:909-920`, `:83` |
| `tracking_mpc_alpha` | **0.0 (disabled)** | `exp(-((α_real-α_cmd)/4.0)^2/0.0625)`, `α_real=(ω_z-ω_z_prev)/dt` | `:923-934`, `:84` |
| `orientation` | **-1.0** | `sum(gravity_xy^2)` | `:937-939`, `:85` |
| `ang_vel_xy` | **-0.3** | `sum(global_ω_xy^2)` | `:941-942`, `:86` |
| `base_height` | **-1.0** | `1-exp(-((h-h_tgt)/0.05)^2)` | `:944-946`, `:87` |
| `posture` | **-1.0** | `sum(w_i*(q_leg-default)^2)*exp(-‖cmd‖²/0.25)`, `w=[1,.5,.5,1,.5,.5]` | `:948-959`, `:88` |
| `torques` | **-1e-4** | `sum(actuator_force^2)` — penalizes the *realized total* torque (post tau_WBC+tau_NN, as measured), not `tau_NN` in isolation | `:961-962`, `:89` |
| `action_rate` | **-0.01** | `sum((a-a_prev)^2)` | `:964-967`, `:90` |
| `dof_pos_limits` | **-1.0** | soft-limit hinge, `soft_factor=0.95` | `:969-973`, `:91` |
| `dof_vel` | **-0.0 (disabled)** | `sum(leg_qvel^2)`, gate multiplied out | `:975-980`, `:92` |
| `termination` | **-100.0** | `done` | `:982-983`, `:93` |

Note: the two MPC-tracking reward terms exist in code (designed to reward the *realized* CoM acceleration/angular-acceleration matching the *MPC's plan*) but are **weight-0, i.e. effectively inert** in this config — so nothing in the reward currently pulls the residual policy toward tracking the MPC's own trajectory; the only coupling to the MPC is via the summed torque and the observation.

Checkpoint evidence (§7) confirms these exact weights and formulas were unchanged between the historical `first_training_residual` run and the current code — no reward-weight drift.

---

## 5. Residual scaling / clipping — exact order of operations

1. **No `jp.clip` on the raw policy action** anywhere in `joystick.py`. Only the policy's own action distribution (`DISTRIBUTION_TYPE="tanh_normal"`, `train_srbd.py:112`, a Brax library concern not independently re-verified from source) implicitly bounds the sampled action.
2. **Scaling is applied to the *target*, not a torque**: `q_des = default_pose + action*0.5`, `dq_des = action*25.0` (`joystick.py:66-67,279-280`). No clip on `action`, `q_des`, or `dq_des`.
3. **PD conversion to torque** (`joystick.py:586-591`): `tau_leg=Kp*(q_des-q)-Kd*qd` (`Kp=50,Kd=1`), `tau_wheel=Kd_wheel*(dq_des-qd)` (`Kd_wheel=0.5`) — unclipped.
4. **Sum**: `ctrl = ctrl_nn + tau` (`joystick.py:593`) — unclipped in Python/JAX.
5. **Only saturation point**: MuJoCo's actuator model inside `mjx.step(...)` (`:596`), via `<motor ctrllimited="true" ctrlrange="-120 120" forcerange="-120 120" gear="1">` on all 8 actuators (`tita.xml:12,16,20,24`) — clamps the *summed* `ctrl` to ±120 N·m.
6. `self._torque_limits` (from `actuator_forcerange[:,1]`, `joystick.py:148`) is computed but **never referenced again** — no explicit software-level soft-limit or residual-specific clip exists.

**Order = scale action→target → PD→torque → sum (unclipped) → saturate only implicitly via MuJoCo's actuator `ctrlrange`.** There is no "clip residual, then add" and no explicit "add, then software-saturate" step — the physics engine is the only place bounds are enforced.

---

## 6. PPO hyperparameters actually used in `train_srbd.py`

Verified directly (`train_srbd.py:111-148`):

```python
POLICY_HIDDEN_LAYER_SIZES = (512, 256, 128)
DISTRIBUTION_TYPE = "tanh_normal"
ZERO_INIT_OUTPUT_LAYER = False
ZERO_INIT_LOAD = True
INIT_STD = 0.03            # only used for SAC path, not PPO (train_srbd.py:1062 commented out for PPO)
NUM_TIMESTEPS = 20_000_000
NUM_EVALS = 10
EPISODE_LENGTH = 1000
NUM_ENVS = 1024

PPO_PARAMS = dict(
    num_timesteps=20_000_000, num_evals=10, reward_scaling=1.0,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=20, num_minibatches=32, num_updates_per_batch=4,
    discounting=0.99, learning_rate=3e-4, entropy_cost=1e-2,
    num_envs=1024, batch_size=256, max_grad_norm=1.0,
    network_factory=dict(
        policy_hidden_layer_sizes=(512,256,128),
        value_hidden_layer_sizes=(512,256,128),
        policy_obs_key="state", value_obs_key="privileged_state"),
    num_resets_per_eval=10, seed=0, deterministic_eval=True,
)
```

- Learning rate `3e-4`. Rollout length (`unroll_length`) `20`. Minibatches `32`. Update epochs (`num_updates_per_batch`) `4`. Discount γ `0.99`. Entropy coef `1e-2`. Grad-clip norm `1.0`. Batch size `256`. Parallel envs `1024`. Total steps `20M`. Episode length `1000` (10 s at `ctrl_dt=0.01`). Network `(512,256,128)` for both actor and critic; asymmetric obs keys `"state"`/`"privileged_state"`.
- **GAE λ and PPO clip ε are NOT set anywhere in `PPO_PARAMS`** — confirmed absent from the dict literal — so they silently take whatever Brax's `ppo.train(...)` default is; this repo does not pin them. (Not independently verified against Brax source — package not present in this sandbox.)
- **Command sampling/curriculum, termination, domain randomization** are identical to what's described in §1/§2 for `joystick.py` (same file, same functions) — `a=[1.0,0.5]`, `b=[0.75,0.75]`, `h=[0.4,0.4]`, no curriculum; termination = upvector flip or base-link floor contact.
- **Domain randomization is defined but NOT applied**: `TitaJoystickFlatTerrain` is registered against `tita_randomize.domain_randomize` (`mujoco_playground/_src/locomotion/__init__.py:175`), but `train_srbd.py`'s `make_envs` (`:177-207`) calls `pg_wrap` (`wrap_for_brax_training`, `:183`) **without** ever calling `registry.get_domain_randomizer(env_name)` (`registry.py:64-69`) or passing a `randomization_fn` into the `train_kwargs` dict fed to `ppo.train` (`train_srbd.py:1335-1342`, no `randomization_fn` key present). A local `wrap_for_brax_training` function at `train_srbd.py:101-106` is itself dead code (never called). **Net effect: as coded, this training run does not apply mass/friction/armature/etc. domain randomization**, despite it existing in the codebase for Tita.

---

## 7. Evidence from checkpoints/ historical runs

Source: `mpx/mpx/examples/checkpoints/TitaJoystickFlatTerrain/saved/first_training_residual/` (residual) and `.../TitaJoystickE2EFlatTerrain/saved/{rl_obs_raw,joystick_with_gitignore}/` (E2E), each with `files_save/copy_train_srbd.txt` (script snapshot actually used) and `metrics_log.csv`/`reward_log.txt`.

**`first_training_residual` hyperparameters** (`files_save/copy_train_srbd.txt:111-147`): identical `PPO_PARAMS` to the current `train_srbd.py` (§6) — `NUM_TIMESTEPS=20M`, `NUM_ENVS=1024`, `unroll_length=20`, `num_minibatches=32`, `num_updates_per_batch=4`, `discounting=0.99`, `lr=3e-4`, `entropy_cost=1e-2`, `batch_size=256`, `max_grad_norm=1.0`. Reward weights (`reward_log.txt:81-99`) also match §4 exactly. **No PPO-hyperparameter or reward-weight drift** between this historical run and the current code. One init-code difference: the saved copy had the custom policy-kernel-init factory **commented out** (`copy_train_srbd.txt:1048`), while the current live file has it **uncommented** (`train_srbd.py:1061`) — a behavior change in weight initialization, though `ZERO_INIT_OUTPUT_LAYER=False` in both so its practical effect is likely small.

**Training outcome (residual run):** `metrics_log.csv` shows reward climbing from **-5.02** (step 0) to **+13.42** (step 16.4M), then plateauing/oscillating **+11.2 to +13.4** through the last logged point **+12.63±2.16** at step 29.49M (`metrics_log.csv` rows 2-11). `avg_episode_length` reaches and holds **1000/1000** (no early termination) from the second eval onward. `reward_log.txt:163` shows a transient **KL spike (`kl=2392.84`)** at eval #7 (step 22.9M), coincident with the run's worst reward-std blowout (`+12.518±2.462`, `:161`) and a following dip at eval #8 (`+11.187±4.730`, `:169`) — the policy recovered by eval #9 without collapsing. Policy std shrank monotonically `0.44→~0.09` (`:114→178`), i.e. normal exploration decay. The final best-checkpoint eval rollout (`evaluation_plots/rollout_info.csv`, 1000 steps) shows **zero terminations**, CoM height held within **0.3966–0.4059 m** of the 0.4 m target, `tracking_lin_vel` reward averaging **0.756/1.0**, `tracking_ang_vel` averaging **0.477/0.5**, torques small (peak actuator force **27.92 N·m**, well under the 120 N·m limit — **no torque saturation** observed), and zero joint-limit violations. **Overall: a stable, successfully-converged run with one transient mid-training instability, not a collapse.**

**Diff — current `train_srbd.py` vs. the saved copy**: only 58 diff lines, all either the kernel-init uncomment noted above, a new unused `ZERO_INIT_LOAD=True` flag (`train_srbd.py:114`), or plotting/diagnostic-only changes. **No PPO/reward changes in `train_srbd.py` itself.**

**Diff — current `joystick.py` vs. the run's saved `copy_joystick.txt`** (424 diff lines) is where the substantive drift lives:
- **Residual target definition changed.** The saved-copy law PD-tracked an action-scaled offset added to the **MPC-integrated** joint target (from `qddot`); the **current** code's `_compute_joint_desired` uses `q_target = self._default_pose` (a **fixed** home pose, confirmed by direct read in §2) — i.e. the network's PD target is no longer coupled to the MPC's own planned trajectory, only to the WBC feedforward torque via the final sum. **This means the residual concept has drifted since the run that is the only clean evidence of successful residual training** — the currently-checked-in code was not the code that produced the `+12.6` converged result.
- **Two new reward terms added but zero-weighted** (`tracking_mpc_accel`, `tracking_mpc_alpha`) — machinery for MPC-trajectory tracking now exists but is inert (matches §4).
- **`command_config.p_stand=0.2` removed** from the config entirely (present in the historical run, gone now).
- **Actor observation composition changed substantially**: historical run's `"state"` was **70-dim** (per its `reward_log.txt` header) vs. current code's **51-dim** (§3) — most `dfcip_*` delta terms were commented out, `joint_pos_des`/`wheel_vel_des` were dropped in favor of `info["mpc_tau"]`, and `mpc_control_fl/fr` changed from delta-vs-previous-step values to raw values. **The current code's observation space does not match what `first_training_residual` was actually trained on.**
- **Command resample countdown likely broken**: `steps_until_next_cmd -= 1` was changed to `-= 0` in a recent edit — effectively disables the natural command-expiry countdown (looks like an in-progress/debug edit, not an intentional final change).
- **MPC command vector** changed `[cmd0,cmd1,cmd2,h]` → `[cmd0, 0.0, cmd1, h]` — the second command slot is now hardcoded to 0 rather than taken from the sampled command.
- `joystickE2E.py` was also modified (36 diff lines) at a timestamp *after* this checkpoint was saved, consistent with the live training process mentioned in the task, simplifying `action_scale_pos/vel` from per-DOF arrays to scalars — self-consistent with §1/§2's scalar description above, but confirms the E2E file is also mid-edit.

**E2E runs for comparison** (`reward_log.txt` in each folder): `rl_obs_raw` (20M steps, 2026-09-01) climbed cleanly **-3.66→+14.23±0.19**, no KL spikes (`kl_mean` 0.03-0.05 throughout), smooth std decay — the cleanest run in the whole checkpoint set. `joystick_with_gitignore` (100M steps, 2026-08-26) rose **-3.65→+14.18±0.85**, plateauing ~13.5-14.3 from ~35M steps, occasional minor early terminations late in training (ep_len 999.3-1000.0). Both used identical PPO hyperparameters/reward weights to the residual run. Other E2E subfolders (one-line notes): `stand_up_first` +14.58 (converged, low-std); `priv_train105` +14.25 (converged); `joystick_height_dummy` +24.24 (converged, different reward scale); `priv_train_0` +14.63 (converged, very low std); `joystick_fixed_height` +24.44 (117.9M steps, converged); `joystick_enhanced` +24.44 (converged); `joystick_refactory` +22.46±3.17, ep_len 991.3 (converged but higher variance); `joystick_first_train` +10.17±3.09 at only 13.1M steps, ep_len 962.7 (early/incomplete run); `joystick_substep_correct` +24.44 (converged). The `+24` vs `+14` reward-scale split across E2E runs suggests a differently-weighted/extended reward config between subsets — not investigated further (out of the requested scope).

**Bottom line on §7:** there is exactly **one** clean successful residual-RL run on disk (`first_training_residual`, converged to ~+12.6, no collapse, no saturation, no sustained early termination). Since that run, `joystick.py`'s residual-target logic, observation composition, and command-config have all changed non-trivially, while `train_srbd.py`'s PPO hyperparameters and reward weights have not. Any new residual run using the current code is therefore not a like-for-like continuation of the one successful precedent — the biggest structural change (decoupling the residual PD target from the MPC-integrated trajectory, in favor of a fixed default pose) directly affects what "residual" means in this codebase today.

---

## 8. Comparison table — E2E vs. residual

| aspect | E2E (`joystickE2E.py`) | Residual (`joystick.py`, current) | Note |
|---|---|---|---|
| Actor obs dim | 34 | 51 | Residual adds MPC plan (`mpc_control_a/acz/alpha/fl/fr`, 9 dims) + `mpc_tau` (8 dims), drops nothing E2E has |
| Critic obs dim | 85 | 102 | Same +17 delta as actor, consistent |
| Action semantics | `q_des=default+a*0.5` (leg), `v_des=a*25` (wheel) | identical formula, but target base is `self._default_pose` in both — **no MPC coupling in the target itself** | Residual's "residual" is only additive at the torque level (§2), not at the target-pose level |
| Torque law | `ctrl = PD(q_des,v_des)` only | `ctrl = PD(q_des,v_des) + tau_WBC`, `tau_WBC` computed with `use_nn=False` (RL-blind) | Confirmed `joystick.py:593` |
| Torque scaling/clip | none in Python; MuJoCo `±120` N·m actuator clamp | identical — none in Python; same MuJoCo clamp | No residual-specific clip exists in either env |
| Reward terms | 10 active (`tracking_lin/ang_vel, orientation, ang_vel_xy, base_height, posture, torques, action_rate, dof_pos_limits, termination`), `dof_vel` disabled | same 10 + 2 more (`tracking_mpc_accel/alpha`) **weight 0**, i.e. same effective reward | No reward difference actually drives residual training toward MPC-plan tracking, despite the machinery existing |
| Weights | identical values to residual (verified, §1 vs §4) | identical | No weight drift, no obvious mistuning in isolation |
| PPO hyperparams | same `PPO_PARAMS` used (per checkpoint evidence, E2E runs used `train_srbd.py` too) | same | Confirmed via checkpoint diff, §7 |
| Domain randomization | defined, wired via registry, **but not invoked by `train_srbd.py`** (§6) — same gap applies to E2E runs launched via this script | same gap | Not residual-specific; a shared limitation of `train_srbd.py` |
| Historical convergence | multiple clean converged runs (`rl_obs_raw` +14.2±0.19 with **no** KL spikes) | one converged run (+12.6±2.16) with a transient KL spike at ~23M steps, and that run used **different residual-target code** than what's checked in now | Residual has less/weaker precedent, and the precedent that exists doesn't match current code |

**What looks mismatched/undertuned in the residual setup relative to what made E2E work:**

1. **Residual target decoupled from the MPC plan.** `_compute_joint_desired` centers the network's PD target on `self._default_pose` (`joystick.py:274`), not on the MPC's own trajectory/qddot integration that the one successful historical run (`first_training_residual`) actually used. The reward has zero-weighted terms (`tracking_mpc_accel/alpha`) that could re-couple the policy to the MPC plan but they are inert (§4). As currently written, "residual" mostly means "PD offset from a fixed home pose, summed with an RL-blind WBC torque" — a materially weaker inductive bias than a true correction-on-top-of-the-plan residual, and different from what converged before.
2. **`tau_WBC` is fully RL-blind by construction** (`use_nn=False` hardcoded, `joystick.py:510`, NN-arg commented out `:509`). The WBC never sees the RL action, so there is no possibility of the WBC adapting its own solve to the learned policy — all adaptation must come from the additive PD term, which only has default-pose as its anchor.
3. **Higher-dimensional, more heterogeneous observation** (51/102 vs 34/85) mixing raw sensor state with MPC planning outputs of very different scales/units (`mpc_control_a`, `mpc_tau` in N·m, contact forces in N) with `normalize_observations=True` as the only scaling defense — more surface area for a harder-to-learn residual mapping than E2E's simpler proprioceptive-only observation.
4. **No `use_only_mpc`/gating actually reachable in `step()`**, despite an `info` key of that name existing elsewhere in `train_srbd.py` — if the intent was ever to warm-start or gate the residual (e.g. MPC-only early in training, blending in the residual later), that mechanism is not wired into the env `step()` and is dead code today.
5. **Command-resample countdown regression** (`steps_until_next_cmd -= 0` instead of `-= 1`, §7) looks like an accidental edit that would make sampled commands never expire via the per-step countdown path — worth checking before the next residual run, since it changes command dynamics relative to both the E2E baseline and the historical residual run.
6. **In-progress `action_scale_pos/vel` experimentation** left as comments (`# prova 0.3`, `# prova 0.15`, per checkpoint evidence) — the checked-in defaults (`0.5`/`25.0`) match history, but the comments signal these were under active tuning, unresolved.
7. **Observation/config drift vs. the only successful precedent** (§7) means whatever is currently training live is not a controlled re-run of `first_training_residual` — if it underperforms or is unstable, the most likely causes (in order of expected impact) are (a) the default-pose-anchored residual target replacing the MPC-integrated one, and (b) the reduced/reshuffled 51-dim observation vs. the historical 70-dim one, rather than PPO hyperparameters or reward weights, which are unchanged.

---

## Open gaps / caveats

- GAE λ and PPO clip ε are not set in `PPO_PARAMS` and take Brax's library default; this repo does not pin them and the audit could not independently verify Brax's default values (package not present in the audit environment).
- `mpc_wrapper_dfcip.py`'s internal QP/FDDP solver math was not traced in full depth beyond confirming the `use_nn` gating and call signature — sufficient to answer the residual-combination question but not a full WBC-algorithm audit.
- The `+24` vs `+14` reward-scale split across historical E2E checkpoint runs was noted but not root-caused (likely a differently-weighted/extended reward config in some runs) — flagged for follow-up if relevant.
- A live training process (`train_srbd.py`, user `jacopo`) was running during this audit; no files were modified and no training was started by this audit.
