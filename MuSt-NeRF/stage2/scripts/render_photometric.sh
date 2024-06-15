#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


export CUDA_VISIBLE_DEVICES=0

SCENE=scannet_room_fewer_cropped
EXPERIMENT=sc_781_00
DATA_DIR=/home/student/Documents/NeRF_tests/multinerf/data/nerf_real_360
CHECKPOINT_DIR=/home/student/Documents/tmp/nerf_results/"$EXPERIMENT"/"$SCENE"



python -m render \
  --gin_configs=configs/photometric.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 60" \
  --gin_bindings="Config.render_dir = '${CHECKPOINT_DIR}/spline_image_name19/'" \
  --gin_bindings="Config.render_video_fps = 10" \
  --gin_bindings="Config.render_spline_keyframes = '${DATA_DIR}/${SCENE}/spline_image_name.txt'"\
  --logtostderr
