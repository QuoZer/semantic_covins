# TODOs
Implement:
- [x] KF counter
- [x] When optim runs after LC detection--> print
- [x] Place recognition on which agent?

Test:
- [x] placerec.cov_consistency_thres: 2

    ran it multiple times:
    - lots of LC at the beginning. There are some later on, but it always stays in suboptimum.
  
- [x] placerec.cov_consistency_thres: 3
    
    ran it 2 times:
    - 2 LC-s and not one in the beginning. Almost realtime till half, then about 1 min at the end
    - lots of LC-s at the beginning. Delayed a lot. Other LC-s throughout. Similar behaviour to previous test (suboptimum).
  
- [x] vis.showkeyframes: 0; cov_consistency_thres: 3

    No noticeable change.

- [ ] COVINS-G params (run COVINS with param values from the COVINS-G (they are updated))
- [ ] placerec.consecutive_loop_dist: moore ; placerec.min_loop_dist: moore
- [ ] clients on different machine


All in all: far from consistent.


### On a new development branch:

Implement:

- [x] Build separation

    - TODO: g20 CMAKELists.txt 78th line checkup (config.h.in)
    - run_docker.sh volume check
    - ORB voc location update for ORB_SLAM

- [x] Only one vocab
- [x] Check if it can work with DBoW only from covins (orb_slam3 and covins both had DoBW2 but they were different versions.)
- [ ] Attila's fixes
- [ ] Webserver cloud publishing to different if
- [ ] Webserver cloud publishing to new thread
- [x] remove install script, and have it in Dockerfile

Test:

- ORB client params
- 

New features
- Dockerfile for jetson
- Add semantics

# Spot COVINS client

This branch aims to prepare the semantic covins project for deployment, including on Spot's hardware. It contains all additions of the `semantic` branch, but organizes it better and includes some Spot related code.   The two `Dockerfile_client` files are intended to be used on (spot) clients. They contain dependencies and sources for the semantic segmentation network, spot wrapper, ORB_SLAM and RealSense. The source volume mounting was commented out for ease of deployment on CORE I/O. The arm64 version of the container was more in focus as it is more relevant to the target hardware, therefore the x86 version lacks some CUDA and PyTorch dependencies. A new launch file was introduced to start all three components of a client. 

# Semantic COVINS 

## Semantic TODO:

- [x] Merge changes from main
- [x] Remove irrelevent dependencies
- Voxblox integration
    - [x] Autostart Voxblox
    - or
    - [ ] Deeper backend Voxblox integration
- [x] Include the semantic segmentation node 
- [X] Integrate mapper into the pipeline 
- [ ] Fully document the changes, parameters and functionality 
- [ ] Make the semantics mode optional (done on the front-end side, needs work on viz side)
- [ ] Clean up the semantic_segmentation container - no need to keep the whole ws 
- Know issues:
    - [ ] The rotation node does not affect all published messages breaking visualizations and mapper. 
    - [ ] Voxblox ESDF is not cleared automatically. Consequitive calls to the post-processing service contribute to the same ESDF unless Voxblox is restarted manually. 
    - [ ] Backend may crash when working on a live map instead of a loaded one. Needs sanity checks on curr_bundle in visualisation_be.cpp 
    - [ ] Mapper is included as a system-wide python package requiring re-installation on any edits.
    - [ ] Many Mapper parameters are not exposed to the interface.

## Semantic usage 

See in `docs/Semantic_COVINS.md`

# COVINS Demo
Copied and updated from readme of covins. Original readme remains in src/covins.


## Docker setup

Install docker.

To build the container, run this in the project folder:

```bash
./build_docker.sh
```

To start the container **(don't forget to update the folder locations at the begining of the `run_docker.sh` file!!)**:

```bash
./run_docker.sh
```

In VSCode you can connect to the container with an other instance of VSCode. You will need the Docker and Dev Containers extensions. After installing those, you will have a docker icon on the left, and if the run_docker.sh script was started you should see a running container. Right clicking on it and you can chose: Attach Visual Studio Code. This opens an instance of Visual Studio Code "inside" the container.

You can setup VSCode to open automatically in a folder with a specific user and extensions. Ctrl+Shift+P and type: 

```
>Dev Containers: Open Attached Container Configuration File
```

After that chose the container. You can set workspace folder and extensions here. And also you can set the user you will connect to the container with. Something like this:

```yaml
{
	"workspaceFolder": "/home/appuser/",
	"remoteUser": "appuser",
	"extensions": [
		"ms-python.isort",
		"ms-python.python",
		"ms-python.vscode-pylance",
		"ms-toolsai.jupyter",
		"ms-toolsai.jupyter-keymap",
		"ms-toolsai.jupyter-renderers",
		"ms-toolsai.vscode-jupyter-cell-tags",
		"ms-toolsai.vscode-jupyter-slideshow",
		"twxs.cmake"
	]
}
```
If you start the container with "Attach Visual Studio Code", the VSCode instance inside the container will start in the /home/appuser folder with the specified extensions and as appuser user.

**You usually need internet when you attach the container first.**

If you need to run something as sudo, you can with the password 'admin'.

This container automatically starts with root user because it starts the webserver. You should setup the remoteUser as above or always switch to appuser by:

```bash
su appuser
```

<a name="setup"></a>
## Basic Setup the classical way (from the original README)
This section explains how you can build the COVINS server back-end, as well as the provided version of the ORB-SLAM3 front-end able to communicate with the back-end. COVINS was developed under Ubuntu *18.04*, and we provide installation instructions for *18.04* as well as *20.04*. Note that we also provide a [Docker implementation](#docker) for simplified deployment of COVINS.

**Note**: Please pay attention to the ```CMAKE_BUILD_TYPE```. Particularly, building parts of the code with ```march=native``` can cause problems on some machines.

<a name="setup_env"></a>
### Environment Setup

#### Dependencies

* ```sudo apt-get update```
* Install dependencies: ```sudo apt-get install libpthread-stubs0-dev build-essential cmake git doxygen libsuitesparse-dev libyaml-cpp-dev libvtk6-dev python3-wstool libomp-dev libglew-dev```
* _catkin_tools_ (from the [catkin_tools manual](https://catkin-tools.readthedocs.io/en/latest/installing.html))
    * ```sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'```
    * ```wget http://packages.ros.org/ros.key -O - | sudo apt-key add -```
    * ```sudo apt-get update```
    * ```sudo apt-get install python3-catkin-tools```
* _ROS_
    * [Melodic](https://wiki.ros.org/melodic/Installation/Ubuntu) (Ubuntu 18)
    * [Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) (Ubuntu 20)

#### Set up your workspace

This will create a workspace for COVINS as ```~/ws/covins_ws```. All further commands will use this  path structure - if you decide to change the workspace path, you will need to adjust the commands accordingly.

* ```git clone https://gitlab.com/gsanya/COVINS_demo.git```
* ```cd COVINS_demo```
* ```catkin init```
* ROS Setup
    * **U18/Melodic**: ```catkin config --extend /opt/ros/melodic/```
    * **U20/Noetic**: ```catkin config --extend /opt/ros/noetic/```
* ```catkin config --merge-devel```
* ```catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo```

<a name="setup_covins"></a>
### COVINS Installation

We provide a script (```covins/install_file.sh```) that will perform a full installation of COVINS, including back-end, front-end, and third-party packages, if the environment is set up correctly. **If the installation fails, we strongly recommend executing the steps in the build script manually one by one**. The script might not perform a correct installation under certain circumstances if executed multiple times.

* ```chmod +x src/covins/install_file.sh```
* ```./src/covins/install_file.sh 8```
    * The argument ```8``` is optional, and specifies the number of jobs the build process should use. 

Generally, when the build process of COVINS or ORB-SLAM3 fails, make sure you have correctly sourced the workspace, and that the libraries in the third-party folders, such as ```DBoW2``` and ```g2o``` are built correctly.

A remark on ```fix_eigen_deps.sh```: compiling code with dependencies against multiple ```Eigen``` versions is usually fatal and must be avoided. Therefore, we specify and download the ```Eigen``` version explicitly through the ```eigen_catkin``` package, and make sure all ```Eigen```  dependencies point to this package.

### Installing ROS Support for the ORB-SLAM3 Front-End

If you want to use `rosbag` files to pass sensor data to COVINS, you need to explicitly build the ORB-SLAM3 front-end with ROS support.

* Install _vision_opencv_:
    * ```cd ~/ws/covins_ws/src```
    * Clone: ```git clone https://github.com/ros-perception/vision_opencv.git```
    * **Check out the correct branch**
        * ```cd vision_opencv/```
        * *U18/Melodic*: ```git checkout melodic```
        * *U20/Noetic*: ```git checkout noetic```
    * Open ```~/ws/covins_ws/src/vision_opencv/cv_bridge/CMakeLists.txt```
        * Add the ```opencv3_catkin``` dependency: change the line ```find_package(catkin REQUIRED COMPONENTS rosconsole sensor_msgs)``` to ```find_package(catkin REQUIRED COMPONENTS rosconsole sensor_msgs opencv3_catkin)```
        * If you are running **Ubuntu 20** (or generally have OpenCV 4 installed): remove the lines that search for an OpenCV 4 version in the ```CMakeLists.txt```. It should look like this:
        ```CMAKE
        set(_opencv_version 3)

        find_package(OpenCV ${_opencv_version} REQUIRED
            COMPONENTS
                opencv_core
                opencv_imgproc
                opencv_imgcodecs
            CONFIG
        )
        ```
    * ```source ~/ws/covins_ws/devel/setup.bash```
    * ```catkin build cv_bridge```
    * [Optional] Check correct linkage:
        * ```cd ~/ws/covins_ws/devel/lib```
        * ```ldd libcv_bridge.so | grep opencv_core```
        * This should only list ```libopencv_core.so.3.4``` as a dependency
* ```catkin build ORB_SLAM3```
* [Optional] Check correct linkage:
    * ```cd ~/ws/covins_ws/src/covins/orb_slam3/Examples/ROS/ORB_SLAM3```
        * ```ldd Mono_Inertial | grep opencv_core```
        * This should mention ```libopencv_core.so.3.4``` as the only ```libopencv_core``` dependency

## Run the demo

You can either run:
- the online demo
- a recording of the running algorithm

### The online demo

You will need the ASUS router, 2 COVINS client boxes with batteries and cables (yellow with black front-plate for holding a phone) numbered 1 and 2 and a laptop/PC to run the server on.

You should plug-in your laptop/PC to the router and give a static IP to it. You can reach the router at router.asus.com and the login is mplab/mplab.

- 192.168.1.10 is for the laptop of Sanya
- 192.168.1.20 is for the ASUS ROG demo laptop

*Please do not delete these!*

The 2 clients automatically connect to the router and are given a static IP. You can ssh into them:

```bash
ssh deva-jnx30d-1@192.168.1.11
ssh deva-jnx30d-4@192.168.1.12
```

The password for the users is deva.

*The IP numbering is based on the 3D printed number on the box, but because of previous hardware failure (jnx30d-3 died and jnx30d-2 doesn't work with external battery for some reason) the number in the username is not consistent with these.*

COVINS and the RealSense ROS camera driver are installed on the automatically mounted SSD of the clients. This SSD is mounted at `/media/deva-jnx30d-x/ssd`. You can find the COVINS folder here.

If you are using your own laptop/PC, you should update the server IP on the clients. This is stored in the COVINS folder in `src/covins/covins_comm/config/config_comm.yaml`. You can update it through ssh using `nano`.

If the IP addresses are set up and you can ssh into both clients you should start the container:

```bash
./run_docker.sh
```

Then start COVINS either by attaching 4 terminals to the container using `docker exec -it containername bash` or by attaching a VSCode instance to the container and opening 4 terminals there. You should use the appuser user account. If you are root you can switch with `su appuser`.

The 4 terminals:

T1 (roscore):
```bash
roscore
```
T2 (backend):
```bash
cd /home/appuser/COVINS_demo
source devel/setup.bash
rosrun covins_backend covins_backend_node
```
T3 (tf transform for visualization):
```bash
cd /home/appuser/COVINS_demo
source devel/setup.bash
roslaunch covins_backend tf.launch
```
T4 (rviz):
```bash
cd /home/appuser/COVINS_demo
source devel/setup.bash
rviz -d /home/appuser/COVINS_demo/src/covins/covins_backend/config/covins.rviz
```

You can start the clients now. Both will need two terminals attached with ssh.

T1 (camera):
```bash
./cam.sh
```
T2 (client):
```bash
./client.sh
```

These scripts cd into the correct directory, source the workspace and launch some rosnodes.

You can view the result of the floorplan extraction in the browser at the IP of the server from any device on the same network.

*The WI-FI password for the ASUS router is: 1qazxsw2.*

### Recording

If you want to record the run, you will need one more terminal for the server and for each client. An example for the rosbag records:

On the server:
```bash
cd /home/appuser/COVINS_demo/bags/
rosbag record -O covins_output_`date +"%Y-%m-%d.%I-%M-%S"` /covins_cloud_be /tf /covins_cloud_filtered /covins_markers_be /covins_trajectories
```

On the clients:
```bash
cd /media/deva-jnx30d-x/ssd
rosbag record -O client_recorded_`date +"%Y-%m-%d.%I-%M-%S"` /camera/imu /camera/infra1/image_rect_raw
```

You can then copy the recorded files from the clients over ssh using `scp`.

If you start the bags together you can re-record everything into one big bag (which can be played back easier):
```bash
rosbag record -O covins_full.bag /covins_cloud_be /tf /covins_cloud_filtered /covins_markers_be /covins_trajectories /client2/camera/imu /client2/camera/infra1/image_rect_raw /client3/camera/imu /client3/camera/infra1/image_rect_raw
```

### Play back a recording

You only need a PC with a built COVINS_demo container and the bag file for it.

When setting up the bag_folder in the `run_docker.sh`, make sure to put the bag file into the attached folder on the host.

Inside the container as appuser:
```bash
cd /home/appuser/COVINS_demo
source devel/setup.bash
roslaunch covins_backend play_recording_x_client.launch
```

*You can run either with 2 or 3 clients. Change x accordingly.*

The launch file will open a configured rviz and an `rqt_bag` instance. You should open your bag inside `rqt_bag`, right-click-->publish all and then start the recording. It should work. 

COVINS is not running in this case, only the floorplan extraction.

You can view the result of the floorplan extraction in the browser at the IP of the server from any device on the same network.

*The WI-FI password for the ASUS router is: 1qazxsw2.*


## Run the demo with tmux

Install tmux and tmuxp on host:
```bash
sudo apt-get install tmux tmuxp
```

If ping 192.168.1.11 and 192.168.1.12 both returns ok than it should work.

From this directory run:
```bash
./run_docker.sh
tmuxp load run_demo_tmux.yaml
```

To kill it: Ctrl+B D
```bash
tmux kill-session"
./run_docker.sh
```
2 enter

### To record:
On window1/pane3:
```bash
rosbag record -O covins_output_`date +"%Y-%m-%d.%I-%M-%S"` /covins_cloud_be /tf /covins_cloud_filtered /covins_markers_be /covins_trajectories
```

On window2/pane2 and pane3:
```bash
./record.sh
```