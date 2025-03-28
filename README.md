# MuSt-NeRF: A Multi-Stage NeRF Pipeline to Enhance Novel View Synthesis 
**✨Best Student Paper Award at the International Conference on Computer Vision Theory and Applications (VISAPP 2025) ✨**

This repository contains the code implementation for MuSt-NeRF, a multi-stage NeRF pipeline designed to enhance novel view synthesis in complex real-world scenes. MuSt-NeRF addresses the limitations of purely photometric approaches by strategically combining geometry-guided and photometry-driven reconstruction techniques. This work was completed as part of a master's thesis at the EAVISE research group, ESAT-PSI, KU Leuven. 

## Directory Structure

- `pre-processing`: Contains scripts to obtain and pre-processing ScanNet data 
- `obtain_poses`: Contains scripts for obtaining camera poses using COLMAP
- `MuSt-NeRF`: Contains the main implementation of the MuSt-NeRF pipeline, including training and evaluation scripts.
- `results_script`: Contains scripts for generating the metrics for the rendered visualizations.

## Setup

To set up the environment for this project, follow these steps:

```
# Clone the repo.
git clone https://github.com/yourusername/MuSt-NeRF.git
cd MuSt-NeRF

# Make a conda environment.
conda create --name mustnerf python=3.9
conda activate mustnerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt
pip install -r requirements_cuda.txt

# Note that this implementation is done with CUDA 12.1 but can also be performed with CUDA 11.
# Modify requirements_cuda accordingly based on the supported hardware

# Install additional dependencies if necessary.
# For example, for COLMAP:
# sudo apt-get install colmap
```

## Running MuSt-NeRF

Follow these steps to run the complete MuSt-NeRF pipeline on a ScanNet scene:

**1. Downloading and Preprocessing ScanNet Data:**
* **Download ScanNet Data:**

   ```bash
   python preprocessing/obtain_scannet/download-scannet.py -o ~/some/directory --id sceneid
   ```

* **Extract RGB and Depth Information:**
   ```bash
   python preprocessing/obtain_scannet/reader.py --filename path_to_sceneid.sens --output_path ./output_path
   ```

* **Crop Images (to remove black pixels):**
   ```bash
   python preprocessing/image_cropper.py 
   ```

* **Resize Images:**
   ```bash
   bash preprocessing/custom_resizer.sh ./path_to_image_folder/
   ```

**2. COLMAP for Pose Estimation:**
   ```bash
   bash obtain_poses/colmap_pipeline.sh 
   # OR
   python obtain_poses/colmap_looper.py
   ```

**3. Stage 1 MuSt-NeRF (Geometry-Based Reconstruction):**

   * **Train:**
     ```bash
     python MuSt-NeRF/stage1/stage1_main/run_nerf.py train --scene_id <sceneid> --data_dir <directory containing the scenes> --depth_prior_network_path <path to depth completion network ckpt> --ckpt_dir <path to write checkpoints>
     ```
   * **Test:**
     ```bash
     python MuSt-NeRF/stage1/stage1_main/run_nerf.py test --expname <experiment name> --data_dir <directory containing the scenes> --ckpt_dir <path to write checkpoints>
     ```
   * **Evaluate Metrics:**
     ```bash
     python results_script/metrics.py
     ```
   * **Render Walkthrough:**
     ```bash
     python results_script/poses_interpolation.py
     python MuSt-NeRF/stage1/stage1_main/run_nerf.py video --expname <experiment name> --data_dir <directory containing the scenes> --ckpt_dir <path to write checkpoints> 
     ```

**7. Stage 2 MuSt-NeRF (Photometric Refinement):**

   * **Train:**
     ```bash
     bash MuSt-NeRF/stage2/scripts/train_photometric.sh
     ```
   * **Test:**
     ```bash
     bash MuSt-NeRF/stage2/scripts/eval_photometric.sh
     ```
   * **Evaluate Metrics:**
     ```bash
     python results_script/metrics.py
     ```
   * **Transform Poses (Align Coordinate Systems):**
     ```bash
     python results_script/poses_interpolation.py
     python results_script/transformation_code.py
     ```
   * **Render Walkthrough:**
     ```bash
     bash MuSt-NeRF/stage2/scripts/render_photometric.sh
     ```

### OOM errors

You may need to reduce the batch size (`Config.batch_size` or `run_nerf.py`) to avoid out of memory errors. If you do this, but want to preserve quality, be sure to increase the number of training iterations and decrease the learning rate by whatever scale factor you decrease batch size by.

### Acknowledgements
We thank [JAXNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) and [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), from which this repository borrows code. 

## Citation
If you use/build on this work, please cite the work as follows:

```
@conference{visapp25,
author={Sudarshan Raghavan Iyengar and Subash Sharma and Patrick Vandewalle},
title={MuSt-NeRF: A Multi-Stage NeRF Pipeline to Enhance Novel View Synthesis},
booktitle={Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 2: VISAPP},
year={2025},
pages={563-573},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0013169100003912},
isbn={978-989-758-728-3},
issn={2184-4321},
}

```   
