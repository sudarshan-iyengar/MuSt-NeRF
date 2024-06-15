import numpy as np

file_path = "spline_pose.txt"
with open(file_path, "r") as file:
    poses_text = file.readlines()

modified_poses_text = []


for pose_text in poses_text:
    if "np.array" in pose_text:
        split_text = pose_text.split("[")
        pose_data = split_text[-1][:-2]
        modified_pose_data = f"{pose_data},\n    [0, 0, 0, 1],\n"
        modified_pose_text = "[".join(split_text[:-1]) + modified_pose_data + "]"
        modified_poses_text.append(modified_pose_text)
    else:
        modified_poses_text.append(pose_text)
with open(file_path, "w") as file:
    for pose_text in modified_poses_text:
        file.write(pose_text)

print("Poses modified and written back to the file.")

