# Spatially Generalizable Mobile Manipulation via Adaptive Experience Selection and Dynamic Imagination

Repository providing the source code for the paper "Spatially Generalizable Mobile Manipulation via Adaptive Experience Selection and Dynamic Imagination", see the [project website](https://307lab.github.io/SG-MM_Web/).  
Please cite the paper as follows:

    @article{SG-MM,
        title={Spatially Generalizable Mobile Manipulation via Adaptive Experience Selection and Dynamic Imagination},
        journal={International Joint Conference on Artificial Intelligence}, 
        author={Ping Zhong and Liangbai Liu and Bolei Chen and Tao Wu},
        year={2026},
    }

<img src=".\images\real_robot.png">
<!-- <img src=".\images\method.jpg"> -->

## Local installation
For development or qualitative inspection of the behaviours in rviz or gazebo it can be easier to install the setup locally.
The following illustrates the main steps to do this for the Husky-UR5. 
As the different robots come with different ROS dependencies, Please install the ROS dependencies required for the corresponding robot. Only partial dependencies for some robots are provided below.

The repository consists of two main parts: a small C++ component connected to python through bindings and the RL agents written in python3.

As not all ROS packages work with python3, the setup relies on running the robot-specific packages in a python2 environment
and our package in a python3 environment.
The environment was tested for Ubuntu 18.04 and ROS melodic.

### Install
Install the appropriate version for your system (full install recommended): http://wiki.ros.org/ROS/Installation

Install the corresponding catkin package for python bindings
        
    sudo apt install ros-[version]-pybind11-catkin

Install moveit and the husky-UR5
    
    sudo apt install ros-[version]-ur-robot-driver ros-[version]-ur-calibration
    sudo apt-get install ros-[version]-moveit
   
Create a catkin workspace (ideally a separate one for each robot)

    mkdir ~/catkin_ws
    cd catkin_ws

Fork the repo and clone into `./src`
    
    cd catkin_ws
    git clone [url] src/sg_mm

Create a python environment. We recommend using conda, which requires to first install Anaconda or Miniconda. Then do

    conda env create -f src/SG-MM.yaml
    conda activate SG-MM

Configure the workspace it to use your environment's python3 (adjust path according to your version)

    catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python3.7m -DPYTHON_LIBRARY=/opt/conda/lib/libpython3.7m.so
    
Build the workspace
    
    catkin build
    
Each new build of the ROS / C++package requires a
    
    source devel/setup.bash
    
To be able to visualise install rviz

    http://wiki.ros.org/rviz/UserGuide
    

## Evaluating on your own task
Tasks are implemented as environment wrappers in `scripts/modulation/envs/tasks.py`.

To construct a new task add a new wrapper to `tasks.py`, add it to ALL_TASKS in `scripts/modulation/envs/__init__.py`.
It can the be used by passing the task name to the `--tasks` flag.


### Run
- If you have a weights and biases account, log into the account. Otherwise first run `wandb offline` in your shell, which will turn off logging.

- Training (set workers etc. as needed, this is the command used in the paper):

        python src/sg_mm/scripts/main_wm.py --load_best_defaults --vis_env --env [husky_ur5/pr2/hsr/tiago] --task picknplace

- Evaluation of a pretrained checkpoint (pick one of [husky_ur5/pr2/hsr/tiago]):

        python src/sg_mm/scripts/eval.py --load_best_defaults --vis_env --env [husky_ur5/pr2/hsr/tiago] --task picknplace 

5. [Only to visualise] start rviz:

        rviz -d src/sg_mm/rviz/rviz_config[_husky].config

Useful command line flags:
- `--load_best_defaults`: use the robot specific best default arguments
- `--vis_env`: publish the visualisation markers to display in rviz
- `--task`: what task to evaluate on


