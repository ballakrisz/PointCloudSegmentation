# Point Cloud Segmentation

## Team information

Team name: Balla Krisztián (solo ha lehet) --> or something  like this (this name choice wasn't the greatest idea)  
Team members: Balla Krisztián (RZWVC0)

## Project description

The goal of this project is to delve into point cloud segmentation using Graph Neural Networks (GNNs). You have to work with the ShapeNet dataset, which contains multiple object categories (from airplanes to wine bottles). By choosing this project, you'll gain expertise in deep learning and spatial data analysis.

## Dataset

The ShapeNetPart dataset will be used, which consists of over 17000 point clouds with both manual and automatic part annotations. It can be found here: https://shapenet.org/download/shapenetsem.  
The related article can be found here: https://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf.


The dataset is also available on https://www.kaggle.com/datasets/mitkir/shapenet/data.

Please note that the dataset will be automatically downloaded during the docker build phase

## Files and folder structure

For the development phase of my solution a docker-based visual studio code approach is taken. To separate the development environment from my own personal environment, a visual studio code server is attached to the running container like so:  
1. Download the 'Dev Containers' and 'Docker' vscode extensions.  
2. On the 'Docker' panel (left sidebar) right click on the running container and choose 'attach visual studio code'

Folder/Files and their purpose:  
1. /misc: contains the requirements.txt file, the .params.example file (more about it later), a .vscode folder (this is just my personal preference) and a .devcontainer folder which includes the neccessary information to install a vscode server and vscode extensions during the docker build.  
2. /src: the main folder, where the actual code goes, it is attached as a volume to the image so no modifications will be lost.  
3. The .gitignore and .dockerignore files are here to (1) act as as safeguard for uploading large or private files to git and (2) to exlude certain files and folder from the docker build context (to make it faster and require less space). 
4. The Dockerfile, build_docker.sh and run_docker.sh are there for the containerization  

Note: For the deployment (finished project), a separate branch will be created with a much more straighforward use

## How to run the project

Clone the repository 
```bash
git clone https://github.com/ballakrisz/PointCloudSegmentation.git
```

Build the docker image
```bash
./build_docker.sh
```

Run the image
```bash
./run_docker.sh
```

To execute the files inside the container
+ follow the method mentioned at the top of the 'Files and folder structure' 
+ inside the container run
```bash
python3 /home/appuser/src/train_segmentation.py
```

Alternatively, if you don't want to attach a vscode server to the container, you can run this shell script, that will split the data into train/test/val splits and visualize 8 point clouds from each split
```bash
./run_docker_no_vscode.sh
``` 

## Related works
PointNet++
github: https://github.com/charlesq34/pointnet2, related paper: https://arxiv.org/abs/1706.02413

## Building and running the project
