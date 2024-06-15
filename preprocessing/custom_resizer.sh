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


# Set to 0 if you do not have a GPU.
USE_GPU=1
# Path to a directory `base/` with images in `base/images/`.
DATASET_PATH=$1


# Resize images.

cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_2

pushd "$DATASET_PATH"/images_2
ls | xargs -P 8 -I {} mogrify -resize 50% {}
popd

cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_4

pushd "$DATASET_PATH"/images_4
ls | xargs -P 8 -I {} mogrify -resize 25% {}
popd

cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_8

pushd "$DATASET_PATH"/images_8
ls | xargs -P 8 -I {} mogrify -resize 12.5% {}
popd
