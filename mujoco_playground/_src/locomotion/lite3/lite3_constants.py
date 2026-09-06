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
"""Defines DeepRobotics Lite3 quadruped constants."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "lite3"
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
      "stairs_terrain": FEET_ONLY_STAIRS_TERRAIN_XML,
      "perlin_terrain": FEET_ONLY_PERLIN_TERRAIN_XML,
  }[task_name]


# Lite3 legs are named front/hind left/right. The ordering below matches the
# joint ordering in lite3.xml (FL, FR, HL, HR x hip_x, hip_y, knee).
FEET_SITES = [
    "FL",
    "FR",
    "HL",
    "HR",
]

FEET_GEOMS = [
    "FL",
    "FR",
    "HL",
    "HR",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]
FEET_FLOOR_FOUND_SENSORS = [f"{geom}_floor_found" for geom in FEET_GEOMS]

NUM_DOFS = 12

ROOT_BODY = "TORSO"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

FLOOR_GEOM = "floor"

TERMINATION_GEOMS = ("TORSO_collision",)
