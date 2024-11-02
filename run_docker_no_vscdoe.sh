#!/bin/bash
#parameters from .param file
source ./misc/.params

#add all the local processes to xhost, so the container reaches the window manager --> cv2.imshow and other display tools won't work without this
xhost + local:

if [[ $( docker ps -a -f name=$container_name | wc -l ) -eq 2 ]];
then
    echo "Stopping it and deleting it... ";
    docker stop $container_name;
    docker rm $container_name;
fi

echo "Starting the container...";
docker run \
    --env DISPLAY=${DISPLAY} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume ${src_folder}:/home/appuser/src \
    --volume ${vscode_folder}:/home/appuser/.vscode \
    --volume ${requirements_file}:/home/appuser/requirements.txt \
    --network host \
    --interactive \
    --privileged \
    --detach \
    --tty \
    --gpus all \
    --runtime=nvidia \
    --name $container_name \
    $image_name:$image_tag \
    python /home/krisz/PointCloudSegmentation/src/seg_models/Pointnet_Pointnet2_pytorch/test_partseg.py \