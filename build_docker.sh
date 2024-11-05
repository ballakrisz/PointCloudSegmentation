#!/bin/bash
#parameters from .param file (please set the use_vscode variable to true or false)
if ! [ -f "misc/.params" ];
then
   echo ".params file not found"
fi
source ./misc/.params
HOST_USER_GROUP_ARG=$(id -g $USER)
VSCODE_COMMIT_HASH=$(code --version | sed -n '2p')
echo $VSCODE_COMMIT_HASH
./misc/download_weights.sh

#build the image
docker build \
    --file Dockerfile\
    --tag $image_name:$image_tag \
    --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG \
    --build-arg VSCODE_COMMIT_HASH=$VSCODE_COMMIT_HASH \
    --build-arg USE_VSCODE=$use_vscode \
    --no-cache \
    .\