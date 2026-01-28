# Tools for Planar Pushing

## Usage
**demo_controller_simple.py**
Test planar pushing.
* Model learning
* Compares planning with DDP vs. Dubin's path.

**demo_controller.py**
Test planar pushing under varying contact model.
* Model learning
* Planning with DDP
* Closed loop control
* Handles modeling learning, planning and control with PushDecision class


## License
This repository is released under the MIT license. 

## Citation
The methods in this repo is an implementation of 
```bibtex
@article{doi:10.1177/0278364919872532,
author = {Jiaji Zhou and Yifan Hou and Matthew T Mason},
title ={Pushing revisited: Differential flatness, trajectory planning, and stabilization},
journal = {The International Journal of Robotics Research},
volume = {38},
number = {12-13},
pages = {1477-1489},
year = {2019},
doi = {10.1177/0278364919872532},
eprint = {https://doi.org/10.1177/0278364919872532},
}
```
This package is created by merging and translating the original matlab-based packages:
* [CloseLoopPushing](https://github.com/yifan-hou/CloseLoopPushing) for model learning, DDP planning & control
* [DDP](https://github.com/yifan-hou/DDP) for implementation of differential dynamic programming
* [PlanarManipulationToolBox](https://github.com/robinzhoucmu/PlanarManipulationToolBox/blob/master/differential_flat/demo_plan_dubin_rect.m) for Dubin's curve computation, and differential flatness of pushing

