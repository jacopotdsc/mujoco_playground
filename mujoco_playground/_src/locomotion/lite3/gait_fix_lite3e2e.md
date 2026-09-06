# Lite3 E2E — dal gait a tre zampe a quello a quattro

Recap del lavoro del 2026-09-06: portare il Lite3 in MuJoCo Playground, farlo
allenare end-to-end, diagnosticare il gait degenerato che ne è uscito e
correggerlo.

Documentazione dell'ambiente: `first_train_lite3e2e.md`, in questa stessa cartella.
Come lanciare le cose: `how_to_run.md`, nella root di `repo_jacopo`.

---

## 1. Integrazione dell'ambiente

Il Lite3 era stato copiato da Aliengo, quindi portava ancora nomi e assunzioni
di quel robot.

### Modello (`xmls/lite3.xml`)

Riscritto per lo standard Playground:

- rimossi `floor` e luce dal file del robot (li fornisce la scena — il nome
  `floor` era duplicato)
- `imu_site` → **`imu`**, più il set completo di sensori che l'ambiente si
  aspetta (`gyro`, `local_linvel`, `accelerometer`, `upvector`, `forwardvector`,
  `global_linvel`, `global_angvel`, `position`, `orientation`)
- aggiunti site e geom di collisione dei piedi **`FL/FR/HL/HR`**, con
  `*_global_linvel`, `*_pos` relativi a `imu`, e i sensori di contatto
- collisioni "feet only" come Aliengo/Go1: tutto inerte tranne i quattro piedi e
  un box `TORSO_collision` come geom di terminazione
- `<option>` (Euler, timestep 0.004), `meshdir`, `armature`/`damping`/`frictionloss`
- keyframe **`home`**: z = 0.31, giunti `[0, −0.8, 1.6]` × 4

### Codice

Rinominati tutti i riferimenti aliengo → lite3 in `base.py`, `joystick.py`,
`joystickE2E.py`, `randomize.py`, `lite3_constants.py`.

Registrati cinque ambienti in `_src/locomotion/__init__.py` e i relativi
iperparametri in `config/locomotion_params.py`. Sono gli unici due file del repo
dove gli ambienti vanno dichiarati: `registry.py`, `locomotion_test.py`,
`train_jax_ppo.py` e `train_rsl_rl.py` iterano su `ALL_ENVS`.

### Bug di consistenza corretti nel portare l'ambiente

1. **Soft limit dei giunti.** `lower * 0.95` funziona solo se il range
   attraversa lo zero. Il ginocchio Lite3 è `[0.524, 2.792]`, tutto positivo: il
   soft-lower finiva *fuori* dal limite hard e la penalità non scattava mai.
   Sostituito col metodo centro/semi-ampiezza usato da `t1` e
   `berkeley_humanoid` → soft `[0.581, 2.735]`.

2. **`_get_reward()` in `reset()`** chiamata con 6 argomenti su una firma da 7
   (mancava `metrics`) → `TypeError`. Lo stesso bug è presente in
   `aliengo/joystick.py`, non toccato.

3. **Guadagni PD.** `Kp = 70`, `Kd = 1.0` ereditati da Aliengo (22 kg) su un
   robot da 11.94 kg. Con `action_scale = 0.5` la coppia a `|action| = 1` era
   **35 N·m contro un `ctrlrange` di ±30**. Portati a `Kp = 35`, `Kd = 0.5`,
   i valori che Playground usa per Go1 (12.7 kg).

4. **`priority` e `friction` sul piede.** Erano in pareggio col `priority="1"`
   del pavimento; a parità MuJoCo fonde con `max()`, quindi la domain
   randomization dell'attrito (`U(0.4, 1.0)`) non avrebbe avuto effetto sotto
   0.6. Rimossi, come in Go1.

5. **`_get_obs(data, info)`** non accettava il terzo argomento `action` che
   l'harness di eval passa quando si usa `--cmd`. Aggiunto come opzionale.

6. **`target_command` ignorato.** `joystickE2E.py` ricampionava direttamente
   `info["command"]` e non leggeva mai `target_command`, mentre
   `joystick.py` (stesso robot) usava lo schema con smoothing. Conseguenza:
   `--cmd` veniva silenziosamente ignorato. Allineati.

7. **Scorciatoie `--name` in `train_srbd.py`** che puntavano a nomi di ambienti
   inesistenti (`LiteE2EJoystickFlatTerrain`). Corrette in
   `litee2e` → `Lite3JoystickE2EFlatTerrain` e `lite3` → `Lite3JoystickFlatTerrain`.

### Primo training

`python train_srbd.py --name litee2e`, 29.5M step, 2h su RTX 2070.

| step | reward | durata episodio |
|---|---|---|
| 0 | 0.41 | 132 |
| 9.8M | 10.89 | 923 |
| 16.4M | 14.47 | 1000 |
| 29.5M | **15.27 ± 0.96** | **1000** |

Curva sana, zero terminazioni. Ma guardando il video la locomozione era
visibilmente sbagliata: zoppicante, asimmetrica, con una zampa posteriore che
non toccava mai terra.

---

## 2. Diagnosi

Rollout deterministiche della policy allenata, 400 step, sette comandi
rappresentativi. Strumento: `mpx/examples/diagnose_gait.py`.

### La zampa è HL, non HR

Il sospetto iniziale era la posteriore destra. I duty factor dicono l'opposto:

| comando | FL | FR | **HL** | HR |
|---|---|---|---|---|
| stand | 0.975 | 0.120 | **0.040** | 0.995 |
| vx 0.8 | 0.685 | 0.230 | **0.022** | 0.650 |
| vx 1.4 | 0.583 | 0.230 | **0.022** | 0.540 |
| yaw 1.0 | 0.860 | 0.182 | **0.028** | 0.863 |

HR era anzi una delle due zampe più piantate a terra. L'inversione nel video si
spiega con la `side_camera`, che guarda da −Y: la sinistra del robot appare a
destra sullo schermo.

### Cause escluse, con i numeri

**Non è il modello.** `mj_forward` in posa home dà i quattro piedi a quota
identica a 1e-4 (`z = 0.0243`, bordo inferiore +0.0023). Ordine
giunti/attuatori/azioni allineato, stessi assi, range, masse e `body_pos`
specchiati. L'unica asimmetria è la `diaginertia` di **FR**_HIP con Ixx/Izz
invertiti (4.47e-4 vs 3.95e-4 su 0.55 kg): irrilevante, e su un'altra zampa.

**Non è il sensore di contatto.** I quattro `*_floor_found` puntano ai geom
9/16/23/30 nell'ordine giusto. Prova definitiva: col comando `vy 0.6` la stessa
HL fa **15 touchdown e 29% di contatto**.

**Non è il foot site disallineato.** `site_xpos == geom_xpos` per tutti e
quattro, la stessa convenzione di Go1.

**Non è il clipping della reward.** La somma grezza è negativa nell'**1.00%**
degli step e il taglio sottrae lo **0.12%** della reward totale.

**HL è davvero in aria:** `air_time` massimo consecutivo **3.90 s su una rollout
di 4.00 s**, quota media del piede 0.095 m contro 0.025 m di FL/HR.

### Causa dimostrata

`feet_clearance` è `Σ |z_piede − 0.1| · √|v_xy|` su **tutti e quattro i piedi a
ogni step**, appoggio compreso. Contributo per piede a `vx 0.8` (scala −2.0):

| piede | \|z − 0.1\| medio | costo scalato |
|---|---|---|
| FL (68% appoggio) | 0.0752 | **−0.0833** |
| FR (23% appoggio) | 0.0424 | −0.0535 |
| **HL (2% appoggio)** | **0.0069** | **−0.0108** |
| HR (65% appoggio) | 0.0723 | −0.0762 |

**La zampa che non lavora paga 7.7 volte meno di quelle che sostengono il
robot.** Il termine premia il piede tenuto esattamente a `max_foot_height`, e un
piede parcheggiato lì lo soddisfa in permanenza; uno che fa un passo vero passa
gran parte del ciclo vicino a terra, dove `|z − 0.1| ≈ 0.078`.

E i due termini che dovrebbero imporre il passo si pagano **solo a
`first_contact`**, che per HL non arriva mai:

```
tracking_lin_vel  +0.9583      feet_clearance  −0.2238
tracking_ang_vel  +0.4926      energy          −0.0810
pose              +0.4670      feet_height     −0.0034   <- solo a first_contact
feet_air_time     +0.0002  <- solo a first_contact
```

`feet_air_time` e `feet_height` sono **tre ordini di grandezza** sotto il
tracking. Non usare una zampa non costava nulla; anzi faceva risparmiare energia
e coppia. Il gait a tre zampe era un ottimo legittimo della reward.

### Confronto con Go1

Nessun checkpoint Go1 disponibile, quindi il confronto è strutturale.
`_cost_feet_clearance`, `_cost_feet_height`, `_reward_feet_air_time`,
`_cost_feet_slip`, le scale di reward, il `reset` randomizzato e la convenzione
del foot site sono **identici**. L'exploit è latente anche in Go1.

Una differenza di pipeline che può contare: `learning/train_jax_ppo.py` passa
`randomization_fn=registry.get_domain_randomizer(...)`, mentre `train_srbd.py`
non lo fa mai. Senza perturbazioni di massa e attrito una strategia asimmetrica
fragile non viene mai messa alla prova.

---

## 3. Correzione

Un solo termine aggiunto in `joystickE2E.py`:

```python
def _cost_feet_air_time_limit(self, air_time, commands):
    # A differenza di feet_air_time / feet_height non aspetta first_contact,
    # quindi un piede che non atterra mai viene addebitato di continuo.
    cmd_norm = jp.linalg.norm(commands)
    excess = jp.clip(air_time - self._config.reward_config.max_air_time, 0.0, None)
    return jp.sum(excess) * (cmd_norm > 0.01)
```

| parametro | prima | dopo | perché |
|---|---|---|---|
| `scales.feet_air_time_limit` | — | **−1.0** | rende il costo della zampa parcheggiata pari al tracking guadagnato |
| `reward_config.max_air_time` | — | **0.5 s** | lo swing normale dura 0.05–0.13 s: 0.5 s è 4–10× e scatta solo su una zampa ferma |

Ricalcolata sulle rollout già registrate, prima di riallenare, la penalità
risultava chirurgica: **−1.42 / −1.49 su HL** (contro un tracking di +1.40 /
+1.47) e **0.000 esatto su FL, FR, HR**. Zero anche a comando nullo e su
`vy_only`, dove HL partecipa davvero.

---

## 4. Verifica: secondo training

Stesso comando, 29.5M step. Reward finale 15.26, episodi 1000/1000 — e già a
9.8M era **avanti** al primo training (12.45 contro 10.89) pur portandosi una
penalità in più.

Rollout deterministiche con gli stessi sette comandi:

| metrica (vx 0.8) | prima | dopo |
|---|---|---|
| **HL touchdown** | **1** | **14** |
| **HL duty factor** | **0.022** | **0.225** |
| **HL max intervallo senza contatto** | **3.90 s** | **0.24 s** |
| HR duty factor | 0.650 | 0.650 |
| spread duty fra le 4 zampe | 0.66 | **0.46** |
| errore vx | 0.065 | **0.049** |
| errore vx a 1.4 m/s | 0.128 | **0.112** |
| dtau rms | 2.813 | **2.795** |
| base_z std | 0.0113 | **0.0103** |
| pitch std | 0.0108 | 0.0162 |

Touchdown su tutti i comandi: prima `FL 17, FR 17, HL 1, HR 17`; dopo
`FL 15, FR 14, HL 14, HR 14`. A `vx 1.4`: prima `19, 18, 4, 18`; dopo
`16, 16, 16, 17`.

**Risolto:** gait a tre zampe eliminato su tutti e sette i comandi, HR tocca
terra regolarmente, tracking migliorato invece che peggiorato, `dtau` e
oscillazione verticale della base leggermente scesi.

**Non risolto:** il chattering delle action è invariato (`d2/d1` 1.38 → 1.36,
inversioni di segno 56.4% → 54.5%). Questo dimostra che la non-fluidità è un
problema **indipendente**, non una conseguenza dell'equilibrio su tre zampe. Il
`pitch_std` è salito del 50%, conseguenza attesa del fatto che ora quattro zampe
fanno il passo invece di due che reggevano il robot come un treppiede.

---

## 5. Tentativo fallito, annullato

Misurato che `action_rate` pesava lo 0.2–0.4% della reward mentre il target
articolare si muoveva **2.5–4.5× più veloce dei giunti stessi**, l'ho alzato da
−0.01 a −0.10 e ho rilanciato. Il training è peggiorato:

```
 6.55M  reward 8.287  len 1000   <- picco
 9.83M  reward 5.984  len  940
13.11M  reward 5.462  len  823
```

Due cali consecutivi con la durata degli episodi in discesa, mentre i due
training precedenti a quel punto salivano in modo monotono. `−0.10` compete col
tracking invece di limarne il rumore. **Fermato e riportato a −0.01**, il valore
di Go1.

Nota per un eventuale tentativo futuro: la leva non sembra essere quella. A
−0.01 la penalità pesa lo 0.2–0.4%, a −0.10 già compete col tracking, e non
esiste un valore intermedio ovvio. Più promettente agire su `sim_dt`/`ctrl_dt` o
filtrare il target articolare.

---

## 6. Stato finale

**Policy:** `mpx/examples/checkpoints/Lite3JoystickE2EFlatTerrain/20260906_175022/params_best.pkl`

**File modificati:**

| file | cosa |
|---|---|
| `lite3/xmls/lite3.xml` | modello riscritto per Playground; `priority`/`friction` del piede rimossi |
| `lite3/xmls/sensor_feet.xml`, `sensor_fullcollision.xml` | nomi dei geom |
| `lite3/lite3_constants.py` | costanti del robot |
| `lite3/base.py`, `randomize.py` | rinomina |
| `lite3/joystickE2E.py` | rinomina + 6 correzioni + `feet_air_time_limit` |
| `lite3/joystick.py` | rinomina + 3 correzioni |
| `_src/locomotion/__init__.py` | registrazione |
| `config/locomotion_params.py` | iperparametri PPO |
| `mpx/examples/train_srbd.py` | scorciatoie `--name` |
| `mpx/examples/mjx_policy_tita.py` | reso compatibile con env non-Tita, tasti per `vy` |
| `mpx/examples/diagnose_gait.py` | **nuovo**, strumento di diagnosi |

**Metriche da tenere d'occhio nei prossimi training:**

1. duty factor delle quattro zampe — spread max−min sotto 0.25 (ora 0.46)
2. max intervallo senza contatto per piede — sotto 0.3 s per tutte ✔
3. touchdown per piede su 400 step — tutte entro ±20% ✔
4. `d2/d1` rms delle action — sotto 1.0 (ora 1.36) ✘
5. inversioni di segno di `d(action)` — sotto 30% (ora 54.5%) ✘
6. `rew_feet_air_time_limit` medio — deve tendere a 0 ✔
7. `err_vx` — non deve peggiorare rispetto a 0.049 a vx 0.8 ✔
