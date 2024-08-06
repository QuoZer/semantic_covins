#!/bin/bash
#parameters
bag_folder="/home/matvei/repos/spot_covins/bags"
source_folder="/home/matvei/repos/spot_covins/src/covins/"
container_name="SPOT_client_arm"
image_name="spot-covins-client"
image_tag="arm64"


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
        --volume ${PWD}/.vscode:/home/appuser/COVINS_demo/.vscode \
        --network host \
        --interactive \
        --tty \
        --detach \
        --gpus all \
        --runtime=nvidia \
        --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
        --env NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics \
        --name $container_name \
        $image_name:$image_tag 
        # --volume ${bag_folder}:/home/appuser/COVINS_demo/bags \
        # --volume ${source_folder}:/home/appuser/COVINS_demo/src/covins/ \
        # /usr/sbin/apachectl -D FOREGROUND
fi