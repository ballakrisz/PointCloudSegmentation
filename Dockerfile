# base image is pytorch with 11.8 CUDA and CUDNN8 (this combination worked well for me)
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# common practice
RUN apt-get update

# set shell to bash, set -c flag so the commands are interpreted as strings by deafult
SHELL ["/bin/bash", "-c"]

# set informative colors for the building process
ENV BUILDKIT_COLORS=run=green:warning=yellow:error=red:cancel=cyan

# start with root user
USER root

# build-time argument given by build_docker.sh
ARG HOST_USER_GROUP_ARG
ARG USE_VSCODE

# create group appuser with id 999
# create grour hostgroup. This is needed so appuser can manipulate the host file without sudo
# create user appuser: home at /home/appuser, default shell is bash, id 999 and add to appuser group  
# set sudo password as admin for user appuser
# add user appuser to the following groups:
#   sudo (admin privis)
#   hostgroup
#   adm (system logging) --> might not be necessary
#   dip (network devices)
# finally, copy a .bashrc file into the container
RUN groupadd -g 999 appuser && \
    groupadd -g $HOST_USER_GROUP_ARG hostgroup && \
    useradd --create-home --shell /bin/bash -u 999 -g appuser appuser && \
    echo 'appuser:admin' | chpasswd && \
    usermod -aG sudo,hostgroup,adm,dip appuser && \
    cp /etc/skel/.bashrc /home/appuser/

# set working directory
WORKDIR /home/appuser

# install basic dependencies for everything (useful for development, may be stripped for deployment)
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noniteractive \
    apt-get install -y \
    git \
    git-lfs \
    build-essential \
    wget \
    curl \
    jq \
    gdb \
    sudo \
    nano \
    net-tools \
    unzip \
    cmake \
    python3-dev

# install gdown
USER root
RUN pip install --no-cache-dir gdown

# donwload tha dataset from google drive
USER appuser
RUN gdown https://drive.google.com/file/d/1CNSXU4naZmISacc5bOBx5FSxnYIkjUNW/view --fuzzy

# unzip the files
USER appuser
RUN unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip && \
    rm shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

# Add c++ package .so to python path so sys.path recognizes it
ENV PYTHONPATH="/home/appuser/src/seg_models/PointCloudSegmentation/openpoints/building_blocks/cpp/pointnet2_batch/build/lib.linux-x86_64-3.10:${PYTHONPATH}"

# copy the requirements.txt into the image
USER root
COPY /misc/requirements.txt .

# install python packages (doesn't need to install python and pip, as it's already installed in base image)
USER appuser
RUN pip install --no-cache-dir -r requirements.txt

# remove the requirements file, because it will be attached as a volume later, so it can be modified from within the container
USER appuser
RUN rm requirements.txt

# Download the weights for the models with gdown
USER appuser
COPY ./misc/download_weights.sh download_weights.sh
RUN ./download_weights.sh

#install vscode server and extensions inside the container (if the use_vscode flag is set to true in the /misc/.params file)
USER root
COPY  --chown=appuser:appuser ./misc/.devcontainer/ /home/appuser/.devcontainer/
USER appuser
ARG VSCODE_COMMIT_HASH
RUN if [ "$USE_VSCODE" = "true" ]; then \
        bash /home/appuser/.devcontainer/preinstall_vscode.sh $VSCODE_COMMIT_HASH /home/appuser/.devcontainer/devcontainer.json; \
    fi
RUN rm -r .devcontainer

# start as appuser
USER appuser