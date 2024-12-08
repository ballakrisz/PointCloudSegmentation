#!/bin/bash

# Create the necessary directories
mkdir -p checkpoints/PointNet2PartSeg
mkdir -p checkpoints/PointCloudSegmentation

# First file ID and target location
FILE_ID_1="1_KoO10yQ6FSL505-dtlh3wbWWLpAqi0P"
TARGET_LOCATION_1="checkpoints/PointNet2PartSeg/2024_11_01.pth"

# Download the first file using gdown
echo "Downloading first file..."
gdown --id $FILE_ID_1 -O $TARGET_LOCATION_1

# Second file ID and target location
FILE_ID_2="1OoXtxHIGmmj3H4IjQpnN44g7yYznNdx9"
TARGET_LOCATION_2="checkpoints/PointCloudSegmentation/pcs_2024-12-08-03-11-51_ckpt_latest.pth"

# Download the second file using gdown
echo "Downloading second file..."
gdown --id $FILE_ID_2 -O $TARGET_LOCATION_2

echo "Download complete!"