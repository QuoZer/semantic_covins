#!/bin/bash
#parameters
bag_folder="/home/${USER}/repos/semantic_covins/bags"
output_folder="/home/${USER}/repos/semantic_covins/output"
source_folder="/home/${USER}/repos/semantic_covins/src/covins/"
script_folder="/home/${USER}/repos/semantic_covins/src/covins/scripts"
container_name="COVINS_demo"
image_name="ros-semantic-covins-demo"
image_tag="latest"

#add all the local processes to xhost, so the container reaches the window manager
xhost + local:

#check if correct directory paths are provided
if [ -z "$bag_folder" ] || [ -z "$source_folder" ];
then
    echo "Please provide all directory paths at the begining of the file."
    exit
fi
if  ! [ -d "$bag_folder" ] || ! [ -d "$source_folder" ];
then
    echo "Please make sure you provide directories that exist."
    exit
fi

#if the repository is not shared promt the user to make it shared
if [[ $( git config core.sharedRepository ) != "true" ]];
then
    echo "The repository is not shared (git config core.sharedRepository)."
    echo "If you will use git at any point you should make it shared because if not the file modifications from inside the container will break git."
    echo "Would you like to make the repo shared?"
    select yn in "Yes" "No"; do
        case $yn in
            Yes )
                echo "Changing repo to shared.";
                git config core.sharedRepository true;
                break;;
            No )
                echo "Leaving it untouched.";
                break;;
        esac
    done
fi

#Set permissions for newly created files. All new files will have the following permissions.
#Important is that the group permissions of the files created are set to read and write and execute.
#We add src as a volume, so we will be able to edit and delete the files created in the container.
setfacl -PRdm u::rwx,g::rwx,o::r ./

#check if container exists --> then we only restart it
if [[ $( docker ps -a -f name=$container_name | wc -l ) -eq 2 ]];
then
    echo "Container already exists. Do you want to restart it or remove it?"
    select yn in "Restart" "Remove"; do
        case $yn in
            Restart )
                echo "Restarting it... If it was started without USB, it will be restarted without USB.";
                docker restart $container_name;
                break;;
            Remove )
                echo "Stopping it and deleting it... You should simply run this script again to start it.";
                docker stop $container_name;
                docker rm $container_name;
                break;;
        esac
    done
else
    echo "Container does not exist. Creating it."
    #The container is started with the apachectl -D FOREGROUND command. This needs sudo, so the end of the dockerfile is in sudo user.
    docker run \
        --env DISPLAY=${DISPLAY} \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --volume $bag_folder:/home/appuser/COVINS_demo/bags \
        --volume ${output_folder}/map_data:/home/appuser/COVINS_demo/src/covins/covins_backend/output/map_data \
        --volume ${source_folder}covins_backend:/home/appuser/COVINS_demo/src/covins/covins_backend \
        --volume ${source_folder}covins_comm/config:/home/appuser/COVINS_demo/src/covins/covins_comm/config \
        --volume ${source_folder}covins_comm/include:/home/appuser/COVINS_demo/src/covins/covins_comm/include \
        --volume ${source_folder}covins_comm/src:/home/appuser/COVINS_demo/src/covins/covins_comm/src \
        --volume ${source_folder}orb_slam3/src:/home/appuser/COVINS_demo/src/covins/orb_slam3/src \
        --volume ${source_folder}orb_slam3/include:/home/appuser/COVINS_demo/src/covins/orb_slam3/include \
        --volume ${source_folder}orb_slam3/Examples:/home/appuser/COVINS_demo/src/covins/orb_slam3/Examples \
        --volume $script_folder:/home/appuser/COVINS_demo/src/covins/scripts \
        --volume ${PWD}/.vscode:/home/appuser/COVINS_demo/.vscode \
        --volume ${source_folder}:/home/appuser/COVINS_demo/src/covins/ \
        --network host \
        --interactive \
        --tty \
        --detach \
        --gpus all \
        --runtime=nvidia \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env NVIDIA_DRIVER_CAPABILITIES=all \
        --privileged \
        --name $container_name \
        $image_name:$image_tag  \
        /usr/sbin/apachectl -D FOREGROUND
fi