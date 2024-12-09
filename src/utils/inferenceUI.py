import os
import gradio as gr
import sys
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Adds `src` to path
from utils.visualizer import visualize_points
from data.load_data import ShapeNetSem
from seg_models.PointCloudSegmentation.UI_inference import InferenceUI
from seg_models.Pointnet_Pointnet2_pytorch.pn_UI_inference import PnInferenceUI

def load_image_gallery(photos_folder):
    """
    Function to load the list of image files from the "PHOTOS_FOLDER" 

    args:
        photos_folder: str: Path to the folder containing the 2D projections
    """
    image_files = [
        os.path.join(photos_folder, f) 
        for f in os.listdir(photos_folder) if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    # Sort them so its easier to find the corresponding point cloud
    image_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))

    return image_files


def show_warning(selection: gr.SelectData):
    """
    Function to fetch the index of the currently active photo in the gallery
    """
    current_pcl = selection.index
    return current_pcl


def perform_inference_on_point_cloud(pc_path, model_choice):
    """
    Function to run inference on the point cloud determined by the selected image

    args:
        pc_path: str: The path to the selected image
        model_choice: str: The model choice for running (PointNet or PCS)
    """
    # Run inference based on the model choice
    if model_choice == "PointNet":
        data = pn_dataset[int(pc_path)]
        fig = pn_inference_ui.inference(data)
    else:
        data = dataset[int(pc_path)]
        fig = inference_ui.inference(data)

    return fig  # Return the plotly figure

# Set up paths
PHOTOS_FOLDER = "/home/appuser/src/pcl_projections"  # Folder containing the 2D projections
image_files = load_image_gallery(PHOTOS_FOLDER)

# Dataset for the PCS network
dataset = ShapeNetSem(
    npoints=2048, 
    split='test', 
    preload=False, 
    use_normals=False,
)

# Dataset for the PointNet network
pn_dataset = ShapeNetSem(
    npoints=2500, 
    split='test', 
    preload=False, 
    use_normals=True,
    pointNet=True
)

# Initialize the inference UIs
inference_ui = InferenceUI()
pn_inference_ui = PnInferenceUI()

current_pcl = None
result_point_cloud = None
with gr.Blocks() as demo:
    # Dropdown menu for selecting the model
    model_selector = gr.Dropdown(choices=["PointNet", "PCS"], label="Select Model", value="PointNet")

    # Gallery for displaying the images
    gallery = gr.Gallery(image_files)

    # Run Inference button
    with gr.Row():
        point_cloud_button = gr.Button("Run Inference")

    # Output for displaying the selected image path (its hidden from the user)
    selected_image = gr.Textbox(label="Selected Image #", visible=False)

    # Output for showing the inference result
    result_point_cloud_plot = gr.Plot(label="Inference Result (3D Point Cloud)")

    # Callback to retrieve the selected photo's index
    gallery.select(fn = show_warning, inputs = None, outputs = selected_image)

    # Callback to run inference on the selected point cloud
    point_cloud_button.click(
        fn=perform_inference_on_point_cloud,
        inputs=[selected_image, model_selector],
        outputs=result_point_cloud_plot
    )

    

demo.launch()
