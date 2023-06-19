
This software was developed by [Manuel Kaufmann](mailto:manuel.kaufmann@inf.ethz.ch), [Velko Vechev](mailto:velko.vechev@inf.ethz.ch) and Dario Mylonopoulos.
The original aitviewer codebase is [here](https://github.com/eth-ait/aitviewer).
I just add a `sloper4d_viwer.py` file to visualize the [SLOPER4D](http://www.lidarhumanmotion.net/data-sloper4d/) dataset.

## Installation
Please install this fork locally 
```commandline
git clone https://github.com/climbingdaily/aitviewer.git
cd aitviewer
pip install -e .
```

## Dependencies
Download the SMPL models to `./data/smpl_models/smpl/`

For more advanced installation and for installing SMPL body models, please refer to the [documentation](https://eth-ait.github.io/aitviewer/parametric_human_models/supported_models.html) .

## Run
```bash
python sloper4d_viewer.py --pkl_file /path/to/the/pkl
```
optional parameters
```bash
--is_global  True    # whether to show the global translation of the person
--pose  'pose'       # the pose param to visualize, 'pose' or 'opt_pose'
--scene_path  /path/to/the/scenemesh
```


## Projects using the aitviewer
A sampling of projects using the aitviewer. Let us know if you want to be added to this list!
- Sun et al., [TRACE: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments](https://www.yusun.work/TRACE/TRACE.html), CVPR 2023
- Fan et al., [ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation](https://arctic.is.tue.mpg.de/), CVPR 2023
- Dong et al., [Shape-aware Multi-Person Pose Estimation from Multi-view Images](https://ait.ethz.ch/projects/2021/multi-human-pose/), ICCV 2021
- Kaufmann et al., [EM-POSE: 3D Human Pose Estimation from Sparse Electromagnetic Trackers](https://ait.ethz.ch/projects/2021/em-pose/), ICCV 2021
- Vechev et al., [Computational Design of Kinesthetic Garments](https://ait.ethz.ch/projects/2022/cdkg/), Eurographics 2021
- Guo et al., [Human Performance Capture from Monocular Video in the Wild](https://ait.ethz.ch/projects/2021/human-performance-capture/index.php), 3DV 2021
- Dong and Guo et al., [PINA: Learning a Personalized Implicit Neural Avatar from a Single RGB-D Video Sequence](https://zj-dong.github.io/pina/), CVPR 2022

## Citation
If you use this software, please cite it as below.
```commandline
@software{Kaufmann_Vechev_aitviewer_2022,
  author = {Kaufmann, Manuel and Vechev, Velko and Mylonopoulos, Dario},
  doi = {10.5281/zenodo.1234},
  month = {7},
  title = {{aitviewer}},
  url = {https://github.com/eth-ait/aitviewer},
  year = {2022}
}
```
