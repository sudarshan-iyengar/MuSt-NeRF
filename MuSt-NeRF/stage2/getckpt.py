import numpy as np
import jax
from flax import serialization
from internal import configs, datasets, train_utils
from jax import random
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
configs.define_common_flags()
jax.config.parse_flags_with_absl()

config = configs.load_config(save_config=False)
PyTree = Any

key = random.PRNGKey(20200823)
_, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)

with open('checkpoint', 'rb') as f:
    encoded_bytes = f.read()
print("Done reading")

# # Deserialize the bytes into a Python object
deserialized_obj = serialization.from_bytes(PyTree, encoded_bytes)

weights = deserialized_obj['params']
nerf_mlp_0_params = weights['params']
prop_mlp = nerf_mlp_0_params['PropMLP_0']
layers=[]

for layer_name, layer_params in prop_mlp.items():
    if "Dense_" in layer_name:
        # Extract the layer number from the layer name
        layer_number = int(layer_name.split("_")[1])
        # Append the layer number and bias array as a tuple
        layers.append((layer_number, layer_params['bias']))
# Sort the layers based on their numbers
layers.sort(key=lambda x: x[0])

# Extract only the bias arrays from the sorted list
sorted_bias_arrays = [bias_array for _, bias_array in layers]

for i, bias_array in enumerate(sorted_bias_arrays):
    print(f"Shape of Dense_{i} bias array:", bias_array.shape)
################################################################
#
# nerf_mlp_0_params = nerf_mlp_0_params['NerfMLP_0']
# layers=[]
# for layer_name, layer_params in nerf_mlp_0_params.items():
#     if "Dense_" in layer_name:
#         # Extract the layer number from the layer name
#         layer_number = int(layer_name.split("_")[1])
#         # Append the layer number and bias array as a tuple
#         layers.append((layer_number, layer_params['bias']))
#
# # Sort the layers based on their numbers
# layers.sort(key=lambda x: x[0])
#
# # Extract only the bias arrays from the sorted list
# sorted_bias_arrays = [bias_array for _, bias_array in layers]

# for i, bias_array in enumerate(sorted_bias_arrays):
#     print(f"Shape of Dense_{i} bias array:", bias_array.shape)
#############################################################################
# for key, value in nerf_mlp_0_params.items():
#     if 'Dense' in key:
#         print("Layer reached")

# for layer_name, layer_params in nerf_mlp_0_params.items():
#     print(f"Layer: {layer_name}")
#     for param_name, param_value in layer_params.items():
#         print(f"Parameter: {param_name}")
#         print(param_value)
#         print()  # Add an empty line for clarity
# Save the weight dictionary to a NumPy file
#np.save('model_weights.npy', weight_dict)
