# Point Cloud Segmentation

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

Clone the repository 
```bash
git clone https://github.com/ballakrisz/PointCloudSegmentation.git && \
cd PointCloudSegmentation
```

Make a copy of the misc/.params.example file and name it .params and fill it out according to your file paths, make sure the ***use_vscode='false'***
```bash
cp misc/.params.example misc/.params
```

Build the docker image
```bash
./build_docker.sh
```

Run the image with the following blueprint
```bash
./run_docker_no_vscode.sh --mode ['train'|'test'] [--batch-size batch_size] [--use-pretrained 'true'|'false'] [--model 'pointnet'|'pcs']
```
#### IMPORTANT
If at any point you cancel the script execution with ctrl+c, don't forget to stop the container by running
```bash
docker stop point_cloud_segmentation 
```

#### Testing
To visualize the trained networks predictions run the following:  
```bash
./run_docker_no_vscode.sh --mode test
```
After this, you can use the DrawIO interface at: http://127.0.0.1:7860/  
You can select the model that you want to use (PointNet or PointCloudSegmentator - my own network).  
Then select an image from the gallery that you would like to segment, and then click Run Inference. The output will be shown at the bottom, with the most important metrics as well.

#### Training
To train the networks run the following command, where --batch-size should be as big as your PC can handle (or what you prefer), the --use-pretrained should be either 'true' or 'false'. Example with batch size of 32 and starting from scratch:
```bash
./run_docker_no_vscode.sh --mode train --batch-size 32 --model pointnet
```
If you want to train the PCS network run the following command (the hyperparameters are adjustable in the seg_models/PointCloudSegmentator/cfg/pcs.yaml file)
```bash
./run_docker_no_vscode.sh --mode train --model pcs
```
After this, you can inspect the training in Tensorboard by opening the following url in your browser:   
http://localhost:6006/
## Baseline model
My baseline model of choice is the PointNet++ architecture because my project focuses on the recent uprising of transformer-based approaches, therefore in the last milestone I will incrementally develop a transformer-based part segmentator with hopes of outpreforming the PointNet++ model.  
The important metrics during training were as follow:  
![Local Image](training_metrics/PointNet++/train_loss.png)
![Local Image](training_metrics/PointNet++/test_acc.png)
![Local Image](training_metrics/PointNet++/test_class_iou.png)
![Local Image](training_metrics/PointNet++/test_instance_iou.png)  
The evaluation on the dest dataset resulted in the following values:  
<div align="center">

<div style="display: flex; justify-content: space-around;">

<table style="margin-right: 20px;">
    <thead>
        <tr>
            <th>Class</th>
            <th>mIoU</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Airplane</td>
            <td>0.821437</td>
        </tr>
        <tr>
            <td>Bag</td>
            <td>0.735315</td>
        </tr>
        <tr>
            <td>Cap</td>
            <td>0.841750</td>
        </tr>
        <tr>
            <td>Car</td>
            <td>0.737776</td>
        </tr>
        <tr>
            <td>Chair</td>
            <td>0.905016</td>
        </tr>
        <tr>
            <td>Earphone</td>
            <td>0.714715</td>
        </tr>
        <tr>
            <td>Guitar</td>
            <td>0.904174</td>
        </tr>
        <tr>
            <td>Knife</td>
            <td>0.877231</td>
        </tr>
        <tr>
            <td>Lamp</td>
            <td>0.842915</td>
        </tr>
        <tr>
            <td>Laptop</td>
            <td>0.946049</td>
        </tr>
        <tr>
            <td>Motorbike</td>
            <td>0.595754</td>
        </tr>
        <tr>
            <td>Mug</td>
            <td>0.948060</td>
        </tr>
        <tr>
            <td>Pistol</td>
            <td>0.798120</td>
        </tr>
        <tr>
            <td>Rocket</td>
            <td>0.575504</td>
        </tr>
        <tr>
            <td>Skateboard</td>
            <td>0.751842</td>
        </tr>
        <tr>
            <td>Table</td>
            <td>0.820963</td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Accuracy</td>
            <td>0.93935</td>
        </tr>
        <tr>
            <td>Class avg accuracy</td>
            <td>0.83077</td>
        </tr>
        <tr>
            <td>Class avg mIoU</td>
            <td>0.80104</td>
        </tr>
        <tr>
            <td>Instance avg mIoU</td>
            <td>0.84371</td>
        </tr>
    </tbody>
</table>

</div>

</div>

## My own model
The training and inference scripts can be found at
```bash
src/seg_models/PointCloudSegmentation/*.py
```
The model and it's building blocks can be found at
```bash
src/seg_models/PointCloudSegmentation/openpoints/build.py
src/seg_models/PointCloudSegmentation/openpoints/segmentation/*.py
src/seg_models/PointCloudSegmentation/openpoints/backbone/pointvit.py, spotr.py
```
The validation script can be found at
```bash
src/seg_models/PointCloudSegmentation/openpoints/utils/validate.py
```
For the details of my model and its design choices please refer to the Documentation.  
Though I couldn't manage to overperform the original PointNet++ Implementation, The results are quite close and with more data (transformer are more likely to benefit from it) and more time for hyperparameter optimization (training just one iteration is almost 16 hours) I belive it has the capability to outperform the PointNet++.  
The important metrics during training:
![Local Image](training_metrics/PointCloudSegmentation/pcs_loss.png)
![Local Image](training_metrics/PointCloudSegmentation/pcs_acc.png)
![Local Image](training_metrics/PointCloudSegmentation/pcs_class_miou.png)
![Local Image](training_metrics/PointCloudSegmentation/pcs_ins_miou.png)  
The evaluation on the dest dataset resulted in the following values:
<div align="center">

<div style="display: flex; justify-content: space-around;">
<table style="margin-right: 20px;">
    <thead>
        <tr>
            <th>Class</th>
            <th>mIoU</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Airplane</td>
            <td>0.782713</td>
        </tr>
        <tr>
            <td>Bag</td>
            <td>0.622064</td>
        </tr>
        <tr>
            <td>Cap</td>
            <td>0.808636</td>
        </tr>
        <tr>
            <td>Car</td>
            <td>0.696074</td>
        </tr>
        <tr>
            <td>Chair</td>
            <td>0.881143</td>
        </tr>
        <tr>
            <td>Earphone</td>
            <td>0.708238</td>
        </tr>
        <tr>
            <td>Guitar</td>
            <td>0.892955</td>
        </tr>
        <tr>
            <td>Knife</td>
            <td>0.804441</td>
        </tr>
        <tr>
            <td>Lamp</td>
            <td>0.805346</td>
        </tr>
        <tr>
            <td>Laptop</td>
            <td>0.950196</td>
        </tr>
        <tr>
            <td>Motorbike</td>
            <td>0.537377</td>
        </tr>
        <tr>
            <td>Mug</td>
            <td>0.881492</td>
        </tr>
        <tr>
            <td>Pistol</td>
            <td>0.760589</td>
        </tr>
        <tr>
            <td>Rocket</td>
            <td>0.423337</td>
        </tr>
        <tr>
            <td>Skateboard</td>
            <td>0.703905</td>
        </tr>
        <tr>
            <td>Table</td>
            <td>0.807279</td>
        </tr>
    </tbody>
</table>


<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Accuracy</td>
            <td>0.92543</td>
        </tr>
        <tr>
            <td>Class avg accuracy</td>
            <td>0.79973</td>
        </tr>
        <tr>
            <td>Class avg mIoU</td>
            <td>0.75842</td>
        </tr>
        <tr>
            <td>Instance avg mIoU</td>
            <td>0.81770</td>
        </tr>
    </tbody>
</table>

</div>

</div>

## Related works

| Model       | GitHub                                             | Article                                                |
|-------------|----------------------------------------------------|--------------------------------------------------------|
| PointNet++:   | [GitHub](https://github.com/charlesq34/pointnet2)   | [Article](https://arxiv.org/abs/1706.02413)            |
| SPoTr       | [GitHub](https://github.com/mlvlab/spotr)           | [Article](https://openaccess.thecvf.com//content/CVPR2023/papers/Park_Self-Positioning_Point-Based_Transformer_for_Point_Cloud_Understanding_CVPR_2023_paper.pdf) |
| PointNeXt   | [GitHub](https://github.com/guochengqian/pointnext) | [Article](https://arxiv.org/pdf/2206.04670v2)          |

