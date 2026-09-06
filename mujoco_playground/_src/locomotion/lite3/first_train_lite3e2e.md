# DeepRobotics Lite3

Ambiente di locomozione per il quadrupede Lite3, portato in MuJoCo Playground
partendo dall'ambiente Aliengo. Il riferimento strutturale è **Go1**, che ha
massa e geometria molto più vicine al Lite3 (12.7 kg contro 11.9 kg).

Aggiornato il 2026-09-06.

---

## Il robot

| | |
|---|---|
| massa totale | 11.94 kg |
| `nq` / `nv` / `nu` | 19 / 18 / 12 |
| gambe | **FL, FR, HL, HR** (hind, non RL/RR) |
| giunti per gamba | HipX (abduzione), HipY (coscia), Knee |
| posa `home` | base a z = 0.31 m, giunti `[0, −0.8, 1.6]` × 4 |
| altezza COM a home | 0.293 m |
| piedi a home | FL/FR `x=±0.182`, HL/HR `x=−0.167`, `y=±0.159`, `z=0.024` |

Range dei giunti: HipX `[−0.523, 0.523]`, HipY `[−2.67, 0.314]`, Knee `[0.524, 2.792]`.

Nota: il ginocchio ha un range **tutto positivo**. È il motivo per cui i soft
limit non si calcolano scalando gli estremi (l'idioma di Aliengo/Go1), ma
restringendo attorno al centro del range — vedi `joystickE2E.py`.

---

## File

| file | contenuto |
|---|---|
| `lite3_constants.py` | percorsi XML, nomi di piedi, sensori, root body, geom di terminazione |
| `base.py` | `Lite3Env`: caricamento XML, accessori ai sensori |
| `joystickE2E.py` | task joystick **end-to-end** (solo rete + PD). È quello funzionante |
| `joystick.py` | variante residual RL su MPC. Vedi avvertenza sotto |
| `randomize.py` | domain randomization (attrito pavimento, masse, armature, `qpos0`) |
| `xmls/lite3.xml` | modello del robot |
| `xmls/scene_*.xml` | scene flat / rough / stairs / perlin |
| `xmls/sensor_feet.xml` | sensori di contatto piede–pavimento |

`getup.py` e `handstand.py` sono eredità di Aliengo, puntano a XML che non
esistono e non sono registrati. Ignorali.

---

## Ambienti registrati

In `mujoco_playground/_src/locomotion/__init__.py`:

- `Lite3JoystickE2EFlatTerrain` → `joystickE2E.py` — **usa questo**
- `Lite3JoystickFlatTerrain` → `joystick.py`
- `Lite3JoystickRoughTerrain` / `StairsTerrain` / `PerlinTerrain` → `joystick.py`

Iperparametri PPO in `mujoco_playground/config/locomotion_params.py`.

> **`joystick.py` (residual MPC) non è utilizzabile così com'è.** Importa
> `mpx.config.config_srbd`, che è cablato su Aliengo: `model_path` punta a
> `mpx/data/aliengo/aliengo.xml`, `mass = 24.637 kg`, inerzia e `p_legs0` di
> Aliengo, `robot_height = 0.35`. Le dimensioni combaciano (12 giunti, 4 piedi)
> quindi gira senza errori, ma il feedforward del whole-body controller è
> calcolato per un robot del doppio del peso e il Lite3 si ribalta. Serve un
> `config_srbd_lite3.py` + `mpx/data/lite3/`, che non esistono.

---

## Modello: scelte da conoscere

**Collisioni.** Tutte disabilitate di default; solo i quattro piedi
(`conaffinity=1`) e il box `TORSO_collision` (`contype=1 conaffinity=1`, geom di
terminazione) collidono col terreno. I geom di collisione delle altre membra
esistono ma sono inerti.

**Contatto piede–pavimento.** Il piede **non** dichiara `priority`, `friction`
né `condim`: li detta il pavimento della scena, che ha `priority="1"`. È la
stessa configurazione di Go1 ed è necessaria perché la domain randomization
dell'attrito abbia effetto — a parità di priorità MuJoCo fonde i due valori con
`max()`, e un attrito campionato sotto quello del piede verrebbe ignorato.

**Foot site.** Il site del piede sta al **centro** della sfera di collisione
(`site_xpos == geom_xpos`), come in Go1. Quindi `foot_z` a contatto vale ~0.022
(il raggio), non 0. Conta per `feet_clearance` e `feet_height`.

**Parametri fisici** (confronto con Go1, che non sono stati copiati):

| | Go1 | Lite3 |
|---|---|---|
| armature | 0.005 | 0.008 |
| damping | 0.5 | 0.3 |
| frictionloss | 0.3 anca / 1.0 ginocchio | 0.1 |
| coppia max | ±23.7 / ±35.55 N·m | ±30 N·m |
| integratore / timestep | Euler / 0.004 | Euler / 0.004 |

`frictionloss = 0.1` è basso rispetto a Go1: primo candidato da rivedere se in
futuro punti al sim2real.

---

## Controllo

PD in coppia calcolato **dentro** l'ambiente, a 500 Hz, non dagli attuatori
MuJoCo (che sono `motor` puri con `gear=1`):

```python
target_pos = default_pose + action * action_scale
tau = Kp * (target_pos - qpos[7:]) + Kd * (-qvel[6:])
```

| parametro | valore | perché |
|---|---|---|
| `Kp` | 35.0 | Lite3 11.94 kg ≈ Go1 12.7 kg, che Playground gira a 35. Aliengo usa 70 ma pesa 22 kg, e 70 × 0.5 = 35 N·m sforerebbe il `ctrlrange` di ±30 |
| `Kd` | 0.5 | stessa classe di massa |
| `action_scale` | 0.5 | come Go1 |
| `ctrl_dt` / `sim_dt` | 0.01 / 0.002 | 100 Hz di controllo, 500 Hz di fisica, 5 substep |

L'ordine delle 12 azioni segue l'ordine degli attuatori, che segue l'ordine dei
giunti: `FL, FR, HL, HR` × `HipX, HipY, Knee` → `qpos[7:19]`, `qvel[6:18]`.

---

## Osservazioni

Actor e critic sono **asimmetrici**.

`state` (48), quello che vede la policy, tutto con rumore:

```
linvel locale 3 | gyro 3 | gravity 3 | (q − default) 12 | dq 12 | last_act 12 | command 3
```

`privileged_state` (123), solo per il critic: `state` più gyro, accelerometro,
gravity, linvel, angvel senza rumore, `q − default`, `dq`, coppie degli
attuatori, contatti (4), velocità dei piedi (12), `feet_air_time` (4), forza
esterna sul torso (3), flag di perturbazione (1).

Comando: `[vx, vy, wz]`, ampiezze `a = [1.5, 0.8, 1.2]`. Il setpoint sta in
`info["target_command"]` e `info["command"]` lo insegue con uno smoothing di
0.05 per step. Questo permette a un harness di eval di fissare il comando
scrivendo `target_command`.

---

## Reward

Identiche a Go1 nell'implementazione e nelle scale, **tranne un termine
aggiunto**:

| termine | scala | |
|---|---|---|
| `tracking_lin_vel` | +1.0 | |
| `tracking_ang_vel` | +0.5 | |
| `pose` | +0.5 | |
| `feet_air_time` | +0.1 | pagata solo a `first_contact` |
| `orientation` | −5.0 | |
| `feet_clearance` | −2.0 | |
| **`feet_air_time_limit`** | **−1.0** | **aggiunto, vedi sotto** |
| `lin_vel_z` | −0.5 | |
| `feet_height` | −0.2 | pagata solo a `first_contact` |
| `feet_slip` | −0.1 | |
| `dof_pos_limits`, `termination`, `stand_still` | −1.0 | |
| `action_rate` | −0.01 | |
| `energy` | −0.001 | |
| `torques` | −0.0002 | |
| `ang_vel_xy` | −0.05 | |

`feet_air_time_limit` penalizza il tempo di volo oltre `max_air_time = 0.5 s`,
per piede e **a ogni step**, senza aspettare `first_contact`. Senza di esso una
zampa che non atterra mai non costa nulla e la policy converge a un gait a tre
zampe. Il perché è documentato in `gait_fix_lite3e2e.md`.

Il reward totale è `clip(sum(terms) * dt, 0, 10000)`. Il clipping inferiore a
zero è la convenzione upstream (go1, t1, aliengo) e misurato non nasconde nulla:
la somma grezza è negativa nello 0–1.5% degli step e il taglio sottrae lo 0.12%
del totale.

---

## Training e valutazione

```bash
cd ~/Desktop/repo_jacopo/mpx/mpx/examples
export PYTHONPATH=~/Desktop/repo_jacopo/mujoco_playground:~/Desktop/repo_jacopo/mpx

python train_srbd.py --name litee2e                  # training, ~2h su RTX 2070
python train_srbd.py --eval --name litee2e --load    # rollout + video + plot
python mjx_policy_tita.py --name litee2e --load      # viewer interattivo
python diagnose_gait.py --name litee2e --load        # metriche di gait
```

`PYTHONPATH` è obbligatorio: nell'env `mjpl` sia `mujoco_playground` sia `mpx`
risolvono di default ad alberi diversi da quelli in `repo_jacopo`. Dettagli in
`how_to_run.md`.

Nel viewer interattivo: frecce su/giù per `vx`, **Home/End o PagSu/PagGiù** per
`vy` (le lettere non funzionano, MuJoCo riserva A–Z per i suoi flag di
visualizzazione), frecce sinistra/destra per `wz`, Space per fermare.

`train_srbd.py` **non** applica la domain randomization: non passa mai
`randomization_fn` a `ppo.train`. Vale per tutti i robot, Aliengo incluso.
`randomize.py` esiste ma resta inutilizzato in training.
