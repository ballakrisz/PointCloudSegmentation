#!/bin/bash
#parameters from .param file
source ./misc/.params

#add all the local processes to xhost, so the container reaches the window manager --> cv2.imshow and other display tools won't work without this
xhost + local:

print_usage() {
    echo "Usage: $0 --mode [train|test] [--batch-size batch_size] [--use-pretrained true|false]"
    echo "  train: Train the model"
    echo "  test: Test the model"
    echo "  batch_size: (Optional) Batch size for training or testing (default: 32 for train, 1 for test)"
    echo "  use_pretrained: (Optional) Use pretrained model or not (default: false)"
}

invalidArgument()
{
    echo ""
    echo "An invalid argument was provided, please use -h|--help to see the available options"
    echo ""
    exit 1 
}

use_pretrained=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) print_usage; exit 0 ;;
        --mode)
            if [[ -n "$2" && "$2" != -* ]]; then  # Check if $2 exists and is not another option
                mode="$2"; shift
            else
                echo "Error: --mode requires a value of train or test."
                exit 1
            fi
            ;;
        --batch-size)
            if [[ -n "$2" && "$2" != -* ]]; then
            batch_size="$2"; shift
            else
                if [[ "$mode" == "train" ]]; then
                    batch_size=32
                elif [[ "$mode" == "test" ]]; then
                    batch_size=1
                else
                    echo "Error: --batch-size requires a value."
                    exit 1
                fi
            fi
            ;;
        --use-pretrained)
            if [[ -n "$2" && "$2" != -* ]]; then
            use_pretrained="$2"; shift
            else
            use_pretrained=false
            fi
            ;;
        *) invalidArgument ;;
    esac
    shift
done


if [[ $( docker ps -a -f name=$container_name | wc -l ) -eq 2 ]];
then
    echo "Stopping it and deleting it... ";
    docker stop $container_name;
    docker rm $container_name;
fi
if [ "$mode" == "train" ]; then

    if [ -n "$batch_size" ]; then
        if ! [[ $batch_size =~ ^[0-9]+$ ]]; then
            echo "Invalid batch size. Please provide a valid number."
            print_usage
            exit 1
        fi
    fi

    if [[ "$use_pretrained" != "true" && "$use_pretrained" != "false" ]]; then
        echo "Invalid value for use_pretrained. Please provide 'true' or 'false'."
        print_usage
        exit 1
    fi

    script_path="/home/appuser/src/seg_models/Pointnet_Pointnet2_pytorch/train_partseg.py --batch_size $batch_size"
    if [ "$use_pretrained" == "true" ]; then
        script_path+=" --use_pretrained"
    fi
    echo "Script is $script_path"
    tensorboard_command="/home/appuser/.local/bin/tensorboard --logdir /home/appuser/src/seg_models/Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg/logs --host 0.0.0.0 --port 6006"
elif [ "$mode" == "test" ]; then

    if [ -n "$batch_size" ]; then
        if ! [[ $batch_size =~ ^[0-9]+$ ]]; then
            echo "Invalid batch size. Please provide a valid number."
            print_usage
            exit 1
        fi
        echo "Evaluation on the whole test dataset with a batch size of $batch_size"
    fi

    script_path="/home/appuser/src/seg_models/Pointnet_Pointnet2_pytorch/test_partseg.py --batch_size $batch_size"
    tensorboard_command="echo 'No tensorboard for testing.'"
else
    echo "Invalid argument."
    print_usage
    exit 1
fi

echo "command: $script_path"

echo "Starting the container...";
docker run \
    --env DISPLAY=${DISPLAY} \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume ${src_folder}:/home/appuser/src \
    --volume ${vscode_folder}:/home/appuser/.vscode \
    --volume ${requirements_file}:/home/appuser/requirements.txt \
    --volume ${checkpoints_folder}:/home/appuser/checkpoints \
    --network host \
    --interactive \
    --privileged \
    --detach \
    --tty \
    --gpus all \
    --runtime=nvidia \
    --name $container_name \
    -p 6006:6006 \
    $image_name:$image_tag \
    bash -c "$tensorboard_command & python $script_path"

docker logs -f $container_name
