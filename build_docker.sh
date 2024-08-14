#export variable for building the image
HOST_USER_GROUP_ARG=$(id -g $USER)
#Release RelWithDebInfo or Debug
COVINS_BUILD_TYPE=RelWithDebInfo
NR_JOBS=4
#build the image
docker build .\
    --tag ros-semantic-covins-demo:latest \
    --build-arg "HOST_USER_GROUP_ARG=${HOST_USER_GROUP_ARG}" \
    --build-arg "COVINS_BUILD_TYPE=${COVINS_BUILD_TYPE}" \
    --build-arg "NR_JOBS=${NR_JOBS}" 