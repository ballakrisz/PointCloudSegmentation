# Point Cloud Segmentation
## TODO
make 2 sections in the how to build part, one if using vscode and one if not using vscode
## Team information

Team name: Balla Krisztián (solo ha lehet) --> or something  like this (this name choice wasn't the greatest idea)  
Team members: Balla Krisztián (RZWVC0)

## Project description

The goal of this project is to delve into point cloud segmentation using Graph Neural Networks (GNNs). You have to work with the ShapeNet dataset, which contains multiple object categories (from airplanes to wine bottles). By choosing this project, you'll gain expertise in deep learning and spatial data analysis.

## Dataset

The ShapeNetPart dataset will be used, which consists of over 17000 point clouds with both manual and automatic part annotations.  
Its available on ShapeNet: https://shapenet.org/download/shapenetsem   
and on kaggle: https://www.kaggle.com/datasets/mitkir/shapenet/data.  
The related article can be found here: https://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf.

Please note that the dataset will be automatically downloaded during the docker build phase

## Files and folder structure

For the development phase of my solution a docker-based visual studio code approach is taken. To separate the development environment from my own personal environment, a visual studio code server is attached to the running container so I can write my code in the vscode IDE inside the container (where all the necessary packages are installed):  
1. Download the 'Dev Containers' and 'Docker' vscode extensions.  
2. On the 'Docker' panel (left sidebar) right click on the running container and choose 'attach visual studio code'

Folder/Files and their purpose:  
1. /misc: contains the requirements.txt file, the .params.example file, a .vscode folder (this is just my personal preference) and a .devcontainer folder which includes the neccessary information to install a vscode server and vscode extensions during the docker build.  
2. /src: the main folder, where the actual code goes, it is attached as a volume to the image so no modifications will be lost.  
3. The .gitignore and .dockerignore files are here to (1) act as as safeguard for uploading large or private files to git and (2) to exlude certain files and folders from the docker build context (to make it faster and require less space). 
4. The Dockerfile, build_docker.sh and run_docker.sh are there for the containerization  

Note: For the deployment (finished project), a separate branch will be created with a much more straighforward use

## Building and running the project

### Option 1: using vscode

Clone the repository 
```bash
git clone https://github.com/ballakrisz/PointCloudSegmentation.git
```

Make a copy of the misc/.params.example file and name it .params and fill it out according to your file paths, make sure the ***use_vscode='true'***
```bash
cp misc/.params.example misc/.params
```

Build the docker image
```bash
./build_docker.sh
```

Run the image
```bash
./run_docker.sh
```

Attach a vscdoe server to the running container according to the top of the 'Files and folder structure' section and inside the container run 
```bash
python3 /home/appuser/src/train_segmentation.py
```

### Option 2: without the vscode IDE

Clone the repository 
```bash
git clone https://github.com/ballakrisz/PointCloudSegmentation.git
```

Make a copy of the misc/.params.example file and name it .params and fill it out according to your file paths, make sure the ***use_vscode='false'***
```bash
cp misc/.params.example misc/.params
```

Build the docker image
```bash
./build_docker.sh
```

Run the image. This will automatically start the train_segmentation.py script, which load the data into 3 splits, preprocesses them and visualizes 8-8 point clouds from each of the dataloaders
```bash
./run_docker_no_vscode.sh
```

## Related works
**PointNet++**  
github: https://github.com/charlesq34/pointnet2, related paper: https://arxiv.org/abs/1706.02413

**SPoTr**  
github: https://github.com/mlvlab/spotr, related paper: https://openaccess.thecvf.com//content/CVPR2023/papers/Park_Self-Positioning_Point-Based_Transformer_for_Point_Cloud_Understanding_CVPR_2023_paper.pdf

**PointNeXt**  
github: https://github.com/guochengqian/pointnext, related paper: https://arxiv.org/pdf/2206.04670v2
