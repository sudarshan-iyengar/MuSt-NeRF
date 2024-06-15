import os
import shutil
import subprocess

import sqlite3

def list_missing_rgb(rgb_dir, sparse_dir):
    expected_files = os.listdir(rgb_dir)
    found_files = []
    for line in open(os.path.join(sparse_dir, "images.txt")):
        for f in expected_files:
            if " " + f in line:
                found_files.append(f)
                break
    print("Missing: ")
    for exp_f in expected_files:
        if exp_f not in found_files:
            print(exp_f)

data_dir = os.path.dirname(os.path.abspath(__file__))
verbose = True
rgb_all_dir = os.path.join(data_dir, "images_all")
rgb_train_dir = os.path.join(data_dir, "images_train")

success = False

# run colmap sfm in a loop on all images (train and test) until all images are successfully registered
while not success:
    # delete previous failed reconstruction
    recon_dir = os.path.join(data_dir, "recon")
    if os.path.exists(recon_dir):
        shutil.rmtree(recon_dir)

    # run colmap with all images creating database db_all.db
    db_all = os.path.join(recon_dir, "db_all.db")
    sparse_dir = os.path.join(recon_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    extract_cmd = "colmap feature_extractor  --database_path {} --image_path {} --ImageReader.single_camera 1 --SiftExtraction.max_image_size 3200 --SiftExtraction.max_num_features 16384 --SiftExtraction.peak_threshold 0.004 --SiftExtraction.edge_threshold 10 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.first_octave -1 --SiftExtraction.num_octaves 4 --SiftExtraction.octave_resolution 3 --SiftExtraction.upright 0 --SiftExtraction.domain_size_pooling 1 --SiftExtraction.dsp_min_scale 0.1 --SiftExtraction.dsp_max_scale 3 --SiftExtraction.dsp_num_scales 10 --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index 0".format(db_all, rgb_all_dir)
    match_cmd = "colmap exhaustive_matcher --database_path {}  --SiftMatching.guided_matching 1".format(db_all)
    mapper_cmd = "colmap mapper --database_path {} --image_path {} --output_path {} --Mapper.min_num_matches 30 --Mapper.abs_pose_min_num_inliers 40 --Mapper.init_max_forward_motion 0.95 --Mapper.filter_min_tri_angle 2 --Mapper.tri_create_max_angle_error 1 --Mapper.tri_continue_max_angle_error 1 --Mapper.multiple_models 0".format(db_all, rgb_all_dir, sparse_dir)
    sparse_dir = os.path.join(sparse_dir, "0")
    convert_cmd = "colmap model_converter --input_path={} --output_path={} --output_type=TXT".format(sparse_dir, sparse_dir)
    colmap_cmds = [extract_cmd, match_cmd, mapper_cmd, convert_cmd]

    number_input_images = len(os.listdir(rgb_all_dir))

    for cmd in colmap_cmds:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        for line in process.stdout:
            if verbose:
                print(line)
        process.wait()

    # check completeness of reconstruction
    number_lines = sum(1 for line in open(os.path.join(sparse_dir, "images.txt")))
    number_reconstructed_images = (number_lines - 4) // 2 # 4 lines of comments, 2 lines per reconstructed image
    print("Expect {} images in the reconstruction, got {}".format(number_input_images, number_reconstructed_images))
    if number_input_images == number_reconstructed_images:
        success = True
    else:
        list_missing_rgb(rgb_all_dir, sparse_dir)

# transform the reconstruction such that z-axis points up
sparse_dir = os.path.join(recon_dir, "sparse", "0")
in_sparse_dir = sparse_dir
out_sparse_dir = os.path.join(recon_dir, "sparse{}".format("_y_down"), "0")
os.makedirs(out_sparse_dir, exist_ok=True)
align_cmd = "colmap model_orientation_aligner --input_path={} --output_path={} --image_path={} --max_image_size={}".format(in_sparse_dir, out_sparse_dir, rgb_all_dir, 640)
in_sparse_dir = out_sparse_dir
out_sparse_dir = os.path.join(recon_dir, "sparse{}".format("_z_up"), "0")
os.makedirs(out_sparse_dir, exist_ok=True)
#/home/student/Documents/colmap_testing/scannet/dense_train
trafo_cmd = "colmap model_transformer --input_path={} --output_path={} --transform_path=/home/student/Documents/colmap_testing/scannet/ddp/y_down_to_z_up.txt".format(in_sparse_dir, out_sparse_dir)
convert_cmd = "colmap model_converter --input_path={} --output_path={} --output_type=TXT".format(out_sparse_dir, out_sparse_dir)
colmap_cmds = [align_cmd, trafo_cmd, convert_cmd]
for cmd in colmap_cmds:
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        if verbose:
            print(line)
    process.wait()

# extract features of train images into database db.db
db = os.path.join(recon_dir, "db.db")
extract_cmd = "colmap feature_extractor  --database_path {} --image_path {} --ImageReader.single_camera 1 --SiftExtraction.max_image_size 3200 --SiftExtraction.max_num_features 16384 --SiftExtraction.peak_threshold 0.004 --SiftExtraction.edge_threshold 10 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.first_octave -1 --SiftExtraction.num_octaves 4 --SiftExtraction.octave_resolution 3 --SiftExtraction.upright 0 --SiftExtraction.domain_size_pooling 1 --SiftExtraction.dsp_min_scale 0.1 --SiftExtraction.dsp_max_scale 3 --SiftExtraction.dsp_num_scales 10 --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index 0".format(db, rgb_train_dir)
process = subprocess.Popen(extract_cmd, shell=True, stdout=subprocess.PIPE)
for line in process.stdout:
    if verbose:
        print(line)
process.wait()

# copy sparse reconstruction from all images
constructed_sparse_train_dir = os.path.join(recon_dir, "constructed_sparse_train", "0")
os.makedirs(constructed_sparse_train_dir, exist_ok=True)
camera_txt = os.path.join(constructed_sparse_train_dir, "cameras.txt")
images_txt = os.path.join(constructed_sparse_train_dir, "images.txt")
points3D_txt = os.path.join(constructed_sparse_train_dir, "points3D.txt")
shutil.copyfile(os.path.join(out_sparse_dir, "cameras.txt"), camera_txt)
open(images_txt, 'a').close()
open(points3D_txt, 'a').close()

# keep poses of the train images in images.txt and adapt their id to match the id in database db.db
train_files = os.listdir(rgb_train_dir)
db_cursor = sqlite3.connect(db).cursor()
name2dbid = dict((n, id)  for n, id in db_cursor.execute("SELECT name, image_id FROM images"))
with open(os.path.join(out_sparse_dir, "images.txt"), 'r') as in_f:
    in_lines = in_f.readlines()
for line in in_lines:
    split_line = line.split(" ")
    line_to_write = None
    if "#" in split_line[0]:
        line_to_write = line
    else:
        for train_file in train_files:
            if " " + train_file in line:
                db_id = name2dbid[train_file]
                split_line[0] = str(db_id)
                line_to_write = " ".join(split_line) + "\n"
                break
    if line_to_write is not None:
        with open(images_txt, 'a') as out_f:
            out_f.write(line_to_write)

# run exaustive matcher and point triangulator on the train images
match_cmd = "colmap exhaustive_matcher --database_path {}  --SiftMatching.guided_matching 1".format(db)
sparse_train_dir = os.path.join(recon_dir, "sparse_train", "0")
os.makedirs(sparse_train_dir, exist_ok=True)
triangulate_cmd = "colmap point_triangulator --database_path {} --image_path {} --input_path {} --output_path {}".format(db, rgb_train_dir, \
    constructed_sparse_train_dir, sparse_train_dir)
convert_cmd = "colmap model_converter --input_path={} --output_path={} --output_type=TXT".format(sparse_train_dir, sparse_train_dir)
colmap_cmds = [match_cmd, triangulate_cmd, convert_cmd]
for cmd in colmap_cmds:
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        if verbose:
            print(line)
    process.wait()
