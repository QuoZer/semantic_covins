#!/bin/bash

#export variable for building the image
HOST_USER_GROUP_ARG=$(id -g $USER)
#Release RelWithDebInfo or Debug
COVINS_BUILD_TYPE=RelWithDebInfo
NR_JOBS=8
#build the image
docker build -f Dockerfile_client.x86 .\
    --tag spot-covins-client:x86 \
    --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG\
    --build-arg COVINS_BUILD_TYPE=$COVINS_BUILD_TYPE


