import numpy as np
import json


P_geometric_2114 = np.array([
    [-0.7123714509693563, -0.09250839244059092, 0.695679195584399, 1.9253719360257684],
    [0.6972486772985742, 0.01945820917044165, 0.716565937272963, 1.8327625352936665],
    [-0.0798253392444442, 0.9955220474275298, 0.050640279027944796, 0.5964943955689052],
    [0.0, 0.0, 0.0, 1.0]
])

P_photometric_2114 = np.array([
    [-0.33021411553816216, -0.12006661412351192, 0.9362385626002, 0.9901120199872138],
    [0.9431401431110052, -0.002020463620147789, 0.33238921188781034, 0.6862957451935066],
    [-0.03801721128703881, 0.9927637815347548, 0.11390682923460706, 1.0022359227082187],
    [0.0, 0.0, 0.0, 1.0]
])

P_photometric_0001 = np.array([
    [-0.7678672388929172, 0.2208526431362231, -0.6013351922636091, -0.6717381941021893],
    [-0.6405729701791318, -0.2547536148346282, 0.7244079414284295, -0.0596076000396414],
    [0.0067950946169084, 0.9414481959982897, 0.3370891884067379, -0.0122454050222896],
    [0.0, 0.0, 0.0, 1.0]
])

P_photometric_0002 = np.array([
    [-0.7707216685530667, 0.2197250495796527, -0.5980877964061764, -0.6684400859841960],
    [-0.6371171097699079, -0.2534395981144019, 0.7279080701201557, -0.0535736280648101],
    [0.0083605060382710, 0.9420664905912054, 0.3353219784683885, -0.0122076741673048],
    [0.0, 0.0, 0.0, 1.0]
])

P_photometric_0003 = np.array([
    [-0.7735626274753021, 0.2185972520872712, -0.5948244301921375, -0.6651419778662030],
    [-0.6336423612635330, -0.2521235338802826, 0.7313898288027340, -0.0475396560899787],
    [0.0099105694024902, 0.9426817941615359, 0.3335458222950880, -0.0121699433123200],
    [0.0, 0.0, 0.0, 1.0]
])

#################### Add any more poses or load from a file

# Due to the use of homogeneous coordinates, the origin in one system can be mapped to its equivalent location in
# another coordinate system. This is often useful

#
T = np.dot(P_geometric_2114, np.linalg.inv(P_photometric_2114))

transformed_poses_photo_to_geo = []

poses_photo = []
for i in range(1, 4):
    # Construct the variable name
    var_name = "P_photometric_{:04d}".format(i)
    # Load the pose variable using locals() dictionary
    pose = locals()[var_name]
    # Append the pose to the list
    poses_photo.append(pose)

for pose_photo in poses_photo:
    transformed_pose = np.dot(T, pose_photo)
    transformed_poses_photo_to_geo.append(transformed_pose)


near = 0.10000000149011612
far = 7.7079997062683109
depth_scaling_factor = 1000.0
fx = 584.0
fy = 584.0
cx = 312.0
cy = 234.0

# Convert numpy array to list for JSON serialization
transformed_poses_photo_to_geo = [pose.tolist() for pose in transformed_poses_photo_to_geo]

# Create JSON structure
json_data = {
    "near": near,
    "far": far,
    "depth_scaling_factor": depth_scaling_factor,
    "frames": []
}

# Populate JSON structure with interpolated poses
for i, pose in enumerate(transformed_poses_photo_to_geo):
    frame_data = {
        "file_path": "",
        "depth_file_path": "",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "transform_matrix": pose
    }
    json_data["frames"].append(frame_data)

# Save JSON to file
with open("transforms_video.json", "w") as f:
    json.dump(json_data, f, indent=4)
