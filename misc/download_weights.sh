#!/bin/bash


FILE_ID="1_KoO10yQ6FSL505-dtlh3wbWWLpAqi0P"
target_location="checkpoints/PointNet2PartSeg/2024_11_01.pth"
mkdir -p checkpoints/PointNet2PartSeg

wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O "${target_location}"

FILE_ID="1OoXtxHIGmmj3H4IjQpnN44g7yYznNdx9"
target_location="checkpoints/PointCloudSegmentation/pcs_2024-12-08-03-11-51_ckpt_latest.pth"
mkdir -p checkpoints/PointCloudSegmentation

wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O "${target_location}"