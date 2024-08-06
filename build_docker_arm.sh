#!/bin/bash

# #export variable for building the image
# HOST_USER_GROUP_ARG=$(id -g $USER)
# #Release RelWithDebInfo or Debug
# COVINS_BUILD_TYPE=RelWithDebInfo
# NR_JOBS=8
# #build the image
# docker build -f Dockerfile_client.arm .\
#     --tag spot-covins-demo:latest \
#     --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG\
#     --build-arg COVINS_BUILD_TYPE=$COVINS_BUILD_TYPE


#export variable for building the image

docker buildx create --use --name multiarchbuilder

mkdir -p prebuilt # this is needed due to limitations in docker buildx
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

#export variable for building the image
HOST_USER_GROUP_ARG=$(id -g $USER)
#Release RelWithDebInfo or Debug
COVINS_BUILD_TYPE=RelWithDebInfo
NR_JOBS=8
docker build .\
    --tag spot-covins-client:arm64 \
    --platform linux/arm64 \
    --file Dockerfile_client.arm \
    --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG \
    --build-arg COVINS_BUILD_TYPE=$COVINS_BUILD_TYPE
    # --progress "plain" \