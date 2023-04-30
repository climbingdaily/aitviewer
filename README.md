
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
