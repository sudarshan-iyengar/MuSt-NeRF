import os
import random
import shutil
import numpy as np
from sklearn.cluster import KMeans


original_folder = "/home/student/Documents/ScanNet/scene0781_00/data/color"

new_folder = "/home/student/Documents/NeRF_tests/stage2/scenes_dir/100_scannet_room_images"
num_images_to_select = 120
image_files = [file for file in os.listdir(original_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]


# randomly picking images if necessary, using KMeans Clustering. Not advised on ScanNetDataset due to blurriness of images
# Can be useful for less manual work on better datasets

num_clusters = min(num_images_to_select, len(image_files))
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(features)

# Select one representative image from each cluster
selected_images = []
for cluster_id in range(num_clusters):
    cluster_images = [image_files[i] for i, c in enumerate(cluster_assignments) if c == cluster_id]
    selected_image = random.choice(cluster_images)
    selected_images.append(selected_image)

# Copy selected images to the new folder
for image in selected_images:
    src = os.path.join(original_folder, image)
    dst = os.path.join(new_folder, image)
    shutil.copyfile(src, dst)

print("Selected images copied to the new folder successfully.")

