# LSeg-TensorRT
This project is an acceleration of the deployment of paper [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546)(ICLR 2022) by using TensorRT. The official code for the paper is [lang-seg](https://github.com/isl-org/lang-seg/tree/main).
# Introduction
![](image/lseg.jpg)
**LSeg**, a novel model for language-driven semantic image segmentation. LSeg uses a text encoder to compute embeddings of descriptive input labels (e.g., ''grass'' or 'building'') together with a transformer-based **image encoder** that computes dense per-pixel embeddings of the input image. The image encoder is trained with a contrastive objective to align pixel embeddings to the text embedding of the corresponding semantic class. The text embeddings provide a flexible label representation in which semantically similar labels map to similar regions in the embedding space (e.g., ''cat'' and ''furry''). This allows LSeg to generalize to previously unseen categories at test time, without retraining or even requiring a single additional training sample. Experimental results demonstrate that LSeg achieves highly competitive zero-shot performance compared to existing zero- and few-shot semantic segmentation methods, and even matches the accuracy of traditional segmentation algorithms when a fixed label set is provided.

## Visualization of inference (FP32)
<table>
    <tr>
        <td><img src="image/chair.jpg" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/seg_image_trt2.png" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/seg_image_trt3.png" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/seg_image_trt4.png" width="100%" style="height: 150px; object-fit: cover;"></td>
    </tr>
    <tr>
        <td><img src="image/car.jpg" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/car_trt3.png" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/car_trt4.png" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/car_trt5.png" width="100%" style="height: 150px; object-fit: cover;"></td>
    </tr>
    <tr>
        <td><img src="image/dog.jpg" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/dog_trt2.png" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/dog_trt1.png" width="100%" style="height: 150px; object-fit: cover;"></td>
        <td><img src="image/dog_trt3.png" width="100%" style="height: 150px; object-fit: cover;"></td>
    </tr>
</table>






# Quick Start
If you don't want to spend the effort to export the ONNX model of CLIP by yourself, you can download the ONNX model of the clip_text part from this link: [clip_onnx](https://github.com/Lednik7/CLIP-ONNX).

```bash
python enginerun.py >./log/run1.log 2>&1
```

## Dependencies installation

To begin on your own machine, clone this repository locally
```bash
git clone https://github.com/vlmaps/vlmaps.git
```
Install requirements:
```bash
$ conda create -n vlmaps python=3.8 -y  # or use virtualenv
$ conda activate vlmaps
$ conda install jupyter -y
$ cd vlmaps
$ bash install.bash
```

## Run Demo
```bash
git checkout demo
```

Start the jupyter notebook
```bash
$ jupyter notebook demo.ipynb
```

# Advanced Usage
Use the `master` branch
```bash
git checkout master
```

## Generate Dataset
To build VLMaps for simulated environments, we manually collected 10 sequences of RGB-D videos among 10 scenes in Habitat simulator with Matterport3D dataset. We provide script and pose meta data to generate the RGB-D videos. Please follow the next few steps to generate the dataset.

### Download Matterpot3D dataset
Please check [Dataset Download](https://niessner.github.io/Matterport/), sign the [Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf), and send to the responsible person to request the Matterport3D mesh for the use in Habitat simulator. The return email will attach a python script to download the data. Copy and paste the script to a file `~/download_mp.py`. Run the following to download the data:

```bash
cd ~
# download the data at the current directory
python2 download_mp.py -o . --task habitat
# unzip the data
unzip v1/tasks/mp3d_habitat.zip
# the data_dir is mp3d_habitat/mp3d
```

### Generate Dataset

* Configure the `config/generate_dataset.yaml`.
    * Change the value for `defaults/data_paths` in `config/generate_dataset.yaml` to `default`.
    * Change the `habitat_scene_dir` and `vlmaps_data_dir` in `config/data_paths/default.yaml` according to the following steps.
    * Change the `habitat_scene_dir` to the downloaded Matterport3D dataset `~/mp3d_habitat/mp3d`.
        ```
        # the structure of the habitat_scene_dir looks like this
        habitat_scene_dir
          |-5LpN3gDmAk7
          |   |-5LpN3gDmAk7.glb
          |   |-5LpN3gDmAk7_semantic.ply
          |   |-...
          |-gTV8FGcVJC9
          |   |-gTV8FGcVJC9.glb
          |   |-gTV8FGcVJC9_semantic.ply
          |   |-...
          |-jh4fc5c5qoQ
          |   |-jh4fc5c5qoQ.glb
          |   |-jh4fc5c5qoQ_semantic.ply
          |   |-...
          ...
        ```
    * Change the `vlmaps_data_dir` to downloaded dataset
        ```
        # the structure of the vlmaps_data_dir looks like this
        vlmaps_data_dir
          |-5LpN3gDmAk7_1
          |   |-poses.txt
          |-gTV8FGcVJC9_1
          |   |-poses.txt
          |-jh4fc5c5qoQ_1
          |   |-poses.txt
          ...
        ```
    * Change `data_cfg.resolution.w` and `data_cfg.resolution.h` to adjust the resolution of the generated rgb, depth, and semantic images.
    * Change `rgb`, `depth`, and `semantic` to `true` to generate corresponding data, and to `false` to ignore corresponding data.
    * Change `camera_height` to change the height of camera relative to the robot base

    ```bash
    # go to <REPO_ROOT>/dataset of this repository
    cd dataset
    # the generated dataset will be in the same
    python generate_dataset.py
    ```

### Collect your own data in Habitat-Sim

Run the following code to collect your own data (for example for scene `gTV8FGcVJC9`):

```python
python dataset/collect_custom_dataset.py scene_names=["gTV8FGcVJC9"]
```

The generated data folder will be `<scene_name>_<id>` under `vlmaps_data_dir` in `config/data_paths/default.yaml` (If you already set the `data_paths` to `default`). The `<scene_name>` is `gTV8FGcVJC9` for the above command, and the `<id>` depends on the existing data folders for this scene. If `gTV8FGcVJC9_1` exists, then the new folder name will be `gTV8FGcVJC9_2`.

## Create a VLMap with the Generated Dataset
* Change the value for `defaults/data_paths` in `config/map_creation_cfg.yaml` to `default`.
* Change the `habitat_scene_dir` and `vlmaps_data_dir` in `config/data_paths/default.yaml` according to the steps in the **Generate Dataset** section above.
* Run the following command to build the VLMap. The code builds a 3D map where each voxel contains the LSeg embedding.
    ```bash
    cd application
    python create_map.py
    ```

    ![](media/vlmaps_process.gif)

### Config the Created VLMap
* Change the scene you want to generate VLMap for by changing `scene_id` (0-9) in `config/map_creation_cfg.yaml`
* Customize the map by changing the parameters in `config/params/default.yaml`
  * Change the resolution of the map by changing `cs` (cell size in meter) and `gs` (grid size)
* Customize the camera pose and base pose by changing `config/vlmaps.yaml`. Change the `pose_info` section.
  * `pose_type` means the type of poses stored in `poses.txt` files. Currently we only support `mobile_base` which means the poses are the poses for the base. But you can implement `camera` if you want.
  * `camera_height` means the camera height relative to the base. Change it if you set different camera height when you generate the dataset.
  * `base2cam_rot` means the row-wise flattened rotation matrix from robot base to the camera coordinate frame (z forward, x right, y down).
  * `base_forward_axis`, `base_left_axis`, `base_up_axis`: your robot base coordinate. They mean what is the coordinate of the forward unit vector [1, 0, 0] projected into your robot base frame, the coordinate of the left unit vector [0, 1, 0] projected into your robot base frame, the coordinate of the upward unit vector [0, 0, 1] projected into your robot base frame.
* Other settings in `config/vlmaps.yaml`
  * `cam_calib_mat` is the flattened camera intrinsics matrix
  * `depth_sample_rate`: we only back project randomly sampled `h * w / depth_sample_rate` pixels at each frame. You can change this to a higher value to increase the mapping speed at the cost of having a sparser point cloud at each frame.

## Index a VLMap
* Change the value for `defaults/data_paths` in `config/map_indexing_cfg.yaml` to `default`.
* Change the `habitat_scene_dir` and `vlmaps_data_dir` in `config/data_paths/default.yaml` according to the steps in the **Generate Dataset** section above.
* Run the following command to index a VLMap you built
    ```bash
    cd application
    python index_map.py
    ```

    ![](media/vlmaps_index.gif)

### Configure the Indexing
* Change the file `config/map_indexing_cfg.yaml`
  * `decay_rate`: set the heatmap decay rate. When it is smaller, the transition of the heat is clearer and covers larger area.
  * `index_2d`: set it to `True` to visualize 2D heatmap. Set it to `False` to visualize 3D heatmap.
  * `init_categories`: set it to `True` to provide a fix list of categories (MatterPort3D categories) to pick from indexing (for details you can check [vlmaps/utils/matterport3d_categories.py](vlmaps/utils/matterport3d_categories.py)). When you query with your text, the code will use GPT to find the nearest category in the fixed list. Set it to `False` to just use your input query for indexing. If you set it to `True`, you need to set an environment variable `OPENAI_KEY` to your OPENAI API tokens by running `OPENAI_KEY=xxx` in your terminal.

## Test Navigation

### Setup OpenAI
In order to test object goal navigation and spatial goal navigation tasks with our method, you need to setup an OpenAI API account with the following steps:
1. [Sign up an OpenAI account](https://openai.com/blog/openai-api), login your account, and bind your account with at least one payment method.
2. [Get you OpenAI API keys](https://platform.openai.com/account/api-keys), copy it.
3. Open your `~/.bashrc` file, paste a new line `export OPENAI_KEY=<your copied key>`, and save the file.


### Run Object Goal Navigation

1. Run object goal navigation. The code will load tasks specified in `<scene_folder>/object_navigation_tasks.json`. The results will be saved in `<scene_folder>/vlmap_obj_nav_results/`. 
    ```bash
    cd application/evaluation
    python evaluate_object_goal_navigation.py
    ```
2. To compute the final metrics, run the following:
    ```bash
    cd application/evaluation
    python compute_object_goal_navigation_metrics.py
    ```
3. Config `config/object_goal_navigation_cfg.json`.
    1. Modify `nav/vis` to `true` to visualize navigation results (POV, topdown trajectory, predicted goal etc.).
    2. Modify `scene_id` to either a number (0~9) or a list `[0,1,3]` to specify which scene to evaluate.

### Run Spatial Goal Navigation
1. Run spatial goal navigation. The code will load tasks specified in `<scene_folder>/spatial_goal_navigation_tasks.json`. The results will be saved in `<scene_folder>/vlmap_spatial_nav_results/`. Modify `nav/vis` to `true` in `config/spatial_goal_navigation_cfg.json` to visualize navigation results (POV, topdown trajectory, predicted goal etc.)
    ```bash
    cd application/evaluation
    python evaluate_spatial_goal_navigation.py
    ```
2. To compute the final metrics, run the following:
    ```bash
    cd application/evaluation
    python compute_spatial_goal_navigation_metrics.py
    ```
3. Config `config/spatial_goal_navigation_cfg.json`.
    1. Modify `nav/vis` to `true` to visualize navigation results (POV, topdown trajectory, predicted goal etc.).
    2. Modify `scene_id` to either a number (0~9) or a list `[0,1,3]` to specify which scene to evaluate.


## Citation

If you find the dataset or code useful, please cite:

```bibtex
@inproceedings{huang23vlmaps,
               title={Visual Language Maps for Robot Navigation},
               author={Chenguang Huang and Oier Mees and Andy Zeng and Wolfram Burgard},
               booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
               year={2023},
               address = {London, UK}
} 
```

## TODO
- [ ] **Refactor Other Mappings**
  - [ ] gradcam_map.py
  - [ ] clip_map.py
  - [ ] gtmap.py
- [ ] **Improve Navigation Stack (Looking for Contributions from the Community)**
  - [ ] the code currently uses `pyvisgraph` to build covisibility graph based on an obstacle map for navigation, which often leads to getting stuck or collisions when the robot navigates at the corner of objects (like the corner of the table). The current solution is to dilate the obstacle map before building the covisibility graph, but this will leads to closing of narrow passages (half-open door becomes closed door). I am happy to discuss solutions to this.
  - [ ] navigation stack on real robot with LiDAR, RGBD camera and other sensors.

## License

MIT License
