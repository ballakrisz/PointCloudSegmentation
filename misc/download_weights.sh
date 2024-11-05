#!/bin/bash


FILE_ID="1_KoO10yQ6FSL505-dtlh3wbWWLpAqi0P"
target_location="/home/krisz/PointCloudSegmentation/checkpoints/PointNet2PartSeg/2024_11_01.pth"
mkdir -p /home/krisz/PointCloudSegmentation/checkpoints/PointNet2PartSeg

wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O "${target_location}"
