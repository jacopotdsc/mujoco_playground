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
"""Constants for Tita."""

from etils import epath

from mujoco_playground._src import mjx_env
import jax.numpy as jp

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "tita"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_flat.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_rough.xml"
)
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
)
FEET_ONLY_STAIRS_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_stairs.xml"
)

FEET_ONLY_PERLIN_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_perlin.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
      "perlin_terrain": FEET_ONLY_PERLIN_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "left_leg_4_site",
    "right_leg_4_site",
]

LEFT_FEET_GEOMS = [
    "left_leg_4_collision",
]

RIGHT_FEET_GEOMS = [
    "right_leg_4_collision",
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base_link"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

TITA_NUM_FEET = 2
TITA_WHEEL_INDICES = jp.array([3, 7])
TITA_LEG_INDICES = jp.array([0, 1, 2, 4, 5, 6])

NUM_DOFS = 8
LEG_DOF_IDS = (0, 1, 2, 4, 5, 6)   # position-controlled
WHEEL_DOF_IDS = (3, 7)  

EET_SITES = ("left_leg_4_site", "right_leg_4_site")
FEET_GEOMS = ("left_leg_4_collision", "right_leg_4_collision")
FEET_TOUCH_SENSORS = ("FL_floor_found", "FR_floor_found")
FEET_FLOOR_FOUND_SENSORS = ("FL_floor_found", "FR_floor_found")

FLOOR_GEOM = "floor"
 
# Geoms che, a contatto con il pavimento, terminano l'episodio
# (equivalente di termination_contact_indices in Isaac: "base").
TERMINATION_GEOMS = ("base_link_collision",)
 
# Geoms penalizzati da _reward_collision (penalised_contact_indices).
COLLISION_GEOMS = ("left_leg_3_collision", "right_leg_3_collision")
