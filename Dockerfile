###Setup base
#
#Base image can be tricky. In my oppinion you should only use a few base images. Complex ones with 
#everything usually have special use cases, an in my experience they take more time to understand, 
#than building one from the ground up.
#The base iamges I suggest you to use:
#- ubuntu: https://hub.docker.com/_/ubuntu
#- osrf/ros:version-desktop-full: https://hub.docker.com/r/osrf/ros
#- nvidia/cuda: https://hub.docker.com/r/nvidia/cuda
#
#We are mostly not space constrained so a little bigger image with everything is usually better,
#than a stripped down version.


FROM osrf/ros:noetic-desktop-full
#this is a small but basic utility, missing from osrf/ros. It is not trivial to know that this is
#missing when an error occurs, so I suggest installing it just to bes sure.
RUN apt-get update && apt-get install -y netbase
#set shell 
SHELL ["/bin/bash", "-c"]
#TODO: make it work
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan
#start with root user
USER root

###Create new user
#
#Creating a user inside the container, so we won't work as root.
#Setting all setting all the groups and stuff.
#
###

#expect build-time argument
ARG HOST_USER_GROUP_ARG
#create group appuser with id 999
#create group hostgroup with ID from host. This is needed so appuser can manipulate the host files without sudo.
#create appuser user with id 999 with home; bash as shell; and in the appuser group
#change password of appuser to admin so that we can sudo inside the container
#add appuser to sudo, hostgroup and all default groups
#copy default bashrc and add ROS sourcing
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,plugdev,video,adm,cdrom,dip,dialout appuser && \
    cp /etc/skel/.bashrc /home/appuser/ && \
    echo "source /opt/ros/noetic/setup.bash" >> /home/appuser/.bashrc

###Install the project
#
#If you install multiple project, you should follow the same 
#footprint for each:
#- dependencies
#- pre install steps
#- install
#- post install steps
#
###

#install dependencies for COVINS demo
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive\
    apt-get install -y\
    libpthread-stubs0-dev\
    git\
    build-essential\
    gdb\
    doxygen\
    libsuitesparse-dev\
    libyaml-cpp-dev\
    python3-wstool\
    libomp-dev\
    libglew-dev\
    wget\
    rsync\
    python3-catkin-tools\
    libtool\
    unzip\
    apache2\
    apache2-utils\
    sshpass\
    python3-pip\
    python3-tk

USER appuser
#create dir
RUN mkdir -p /home/appuser/COVINS_demo/src/covins
#copy only those files that are needed for the dependencies
COPY --chown=appuser:appuser ./src/covins/dependencies.rosinstall /home/appuser/COVINS_demo/src/covins/dependencies.rosinstall
COPY --chown=appuser:appuser ./src/covins/fix_eigen_deps.sh /home/appuser/COVINS_demo/src/covins/fix_eigen_deps.sh
#args for build
ARG COVINS_BUILD_TYPE
ARG NR_JOBS
#clone dependencies 
RUN cd /home/appuser/COVINS_demo &&\
    catkin init &&\
    catkin config --extend /opt/ros/noetic/ &&\
    catkin config --merge-devel &&\
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=$COVINS_BUILD_TYPE &&\
    cd src && \
    wstool init && \
    wstool merge covins/dependencies.rosinstall && \
    wstool up && \
    chmod +x covins/fix_eigen_deps.sh && \
    ./covins/fix_eigen_deps.sh && \
    cd vision_opencv && \
    git checkout noetic && \
    cd cv_bridge && \
    sed -i '4d' CMakeLists.txt && \
    sed -i '3afind_package(catkin REQUIRED COMPONENTS rosconsole sensor_msgs opencv3_catkin)' CMakeLists.txt && \
    sed -i '20d' CMakeLists.txt && \
    sed -i '20d' CMakeLists.txt && \
    sed -i '20d' CMakeLists.txt && \
    sed -i '20d' CMakeLists.txt && \
    sed -i '20d' CMakeLists.txt && \
    sed -i '20d' CMakeLists.txt && \
    sed -i '19aset(_opencv_version 3)' CMakeLists.txt

#install dependencies
#build catkin dependencies before copying files
RUN cd /home/appuser/COVINS_demo && \
    catkin build -j2 \
    opengv

RUN cd /home/appuser/COVINS_demo && \
    catkin build -j$NR_JOBS\
    eigen_catkin\
    opencv3_catkin\
    catkin_simple\
    doxygen_catkin\
    gflags_catkin\
    protobuf_catkin\
    suitesparse\
    yaml_cpp_catkin\
    glog_catkin\
    eigen_checks\
    ceres_catkin\
    minkindr\
    minkindr_conversions\
    aslam_cv_common\
    aslam_cv_cameras\
    robopt_open
#install cv_bridge
RUN cd /home/appuser/COVINS_demo && \
    catkin build -j$NR_JOBS cv_bridge
#install pangolin
RUN mkdir -p /home/appuser/COVINS_demo/build_thirdparty/pangolin/lib && \
    mkdir -p /home/appuser/COVINS_demo/build_thirdparty/pangolin/bin && \
    cd /home/appuser/COVINS_demo/build_thirdparty/pangolin && \
    source /home/appuser/COVINS_demo/devel/setup.bash && \
    cmake -DCMAKE_INSTALL_PREFIX=/home/appuser/COVINS_demo/build_thirdparty/pangolin /home/appuser/COVINS_demo/src/pangolin -DCMAKE_BUILD_TYPE=$COVINS_BUILD_TYPE && \
    make -j$NR_JOBS && \
    make install
#copy and unzip voc file
COPY --chown=appuser:appuser ./src/covins/orb_slam3/Vocabulary/ORBvoc.txt.tar.gz /home/appuser/COVINS_demo/src/covins/orb_slam3/Vocabulary/ORBvoc.txt.tar.gz
RUN mkdir /home/appuser/COVINS_demo/voc && \
    cd /home/appuser/COVINS_demo/voc && \
    tar -xf /home/appuser/COVINS_demo/src/covins/orb_slam3/Vocabulary/ORBvoc.txt.tar.gz
#copy and install DBoW2
COPY --chown=appuser:appuser ./src/covins/covins_backend/thirdparty/DBoW2 /home/appuser/COVINS_demo/src/covins/covins_backend/thirdparty/DBoW2   
RUN mkdir -p /home/appuser/COVINS_demo/build_thirdparty/DBoW2/lib && \
    cd /home/appuser/COVINS_demo/build_thirdparty/DBoW2 && \
    source /home/appuser/COVINS_demo/devel/setup.bash && \
    cmake /home/appuser/COVINS_demo/src/covins/covins_backend/thirdparty/DBoW2 -DCMAKE_BUILD_TYPE=$COVINS_BUILD_TYPE && \
    make -j$NR_JOBS
#copy and install g2o
COPY --chown=appuser:appuser ./src/covins/orb_slam3/Thirdparty/g2o /home/appuser/COVINS_demo/src/covins/orb_slam3/Thirdparty/g2o
RUN mkdir -p /home/appuser/COVINS_demo/build_thirdparty/g2o/lib && \
    mkdir -p /home/appuser/COVINS_demo/build_thirdparty/g2o/bin && \
    cd /home/appuser/COVINS_demo/build_thirdparty/g2o && \
    source /home/appuser/COVINS_demo/devel/setup.bash && \
    cmake -DPROJECT_INSTALL_DIR=/home/appuser/COVINS_demo/build_thirdparty/g2o /home/appuser/COVINS_demo/src/covins/orb_slam3/Thirdparty/g2o -DCMAKE_BUILD_TYPE=$COVINS_BUILD_TYPE && \
    make -j$NR_JOBS

#copy and install covins_backend
COPY --chown=appuser:appuser ./src/covins/covins_backend /home/appuser/COVINS_demo/src/covins/covins_backend
COPY --chown=appuser:appuser ./src/covins/covins_comm /home/appuser/COVINS_demo/src/covins/covins_comm
RUN cd /home/appuser/COVINS_demo && \
    catkin build -j$NR_JOBS covins_backend
#copy and install ORB_SLAM3
COPY --chown=appuser:appuser ./src/covins /home/appuser/COVINS_demo/src/covins
RUN mkdir -p /home/appuser/COVINS_demo/build_thirdparty/orb_slam3 && \
    cd /home/appuser/COVINS_demo/build_thirdparty/orb_slam3 && \
    source /home/appuser/COVINS_demo/devel/setup.bash && \
    cmake -DPROJECT_INSTALL_DIR=/home/appuser/COVINS_demo/build_thirdparty/orb_slam3 /home/appuser/COVINS_demo/src/covins/orb_slam3 -DCMAKE_BUILD_TYPE=$COVINS_BUILD_TYPE && \
    make -j$NR_JOBS
#install ORB_SLAM3 ROS
RUN cd /home/appuser/COVINS_demo && \
    catkin build -j$NR_JOBS ORB_SLAM3
#delete source dirs so that we can connect them as a volume.
RUN rm -r /home/appuser/COVINS_demo/src/covins

#install voxblox
RUN cd /home/appuser/COVINS_demo/src && \
    git clone https://github.com/ethz-asl/voxblox.git && \
    wstool merge -t . ./voxblox/voxblox_https.rosinstall && \
    wstool update && \
    catkin build -j$NR_JOBS voxblox_ros 

RUN pip3 install scikit-image \
                 open3d

#install ros2
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

USER root
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y ros-foxy-desktop


#install and configure apache
RUN mkdir -p /var/www/pointmap.net/html
COPY ./web/pointmap.net.conf /etc/apache2/sites-available/pointmap.net.conf
COPY ./web/pointmap.net /var/www/pointmap.net/html 
RUN chown -R appuser:appuser /var/www/pointmap.net/html && \
    chmod -R 755 /var/www/pointmap.net && \
    echo "ServerName pointmap.net" > /etc/apache2/conf-available/servername.conf
EXPOSE 80
RUN apache2ctl start && \
    a2ensite pointmap.net.conf && \
    a2dissite 000-default.conf && \
    a2enconf servername && \
    service apache2 reload
#copy debug configs
COPY --chown=appuser:appuser ./.vscode /home/appuser/COVINS_demo/.vscode