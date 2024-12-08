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

# Function to load the list of image files from the "photos" folder
def load_image_gallery(photos_folder):
    image_files = [
        os.path.join(photos_folder, f) 
        for f in os.listdir(photos_folder) if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    image_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))

    return image_files


def show_warning(selection: gr.SelectData):
    current_pcl = selection.index
    return current_pcl
    #gr.Warning(f"Your choice is #{selection.index}, with image: {selection.value['image']['path']}!")
    #return current_pcl

def perform_inference_on_point_cloud(pc_path):
    data = dataset[int(pc_path)]
    fig = inference_ui.inference(data)
    # pcl, acc, best_part_iou, worst_part_iou, avg_part_iou = data
    # xyz, features, obj_class, labels = data
    # num_points = 100
    # points = np.random.rand(num_points, 3)  # Shape (100, 3) -> 100 3D points

    # # Create a matplotlib figure for 3D plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_axis_off()

    return fig  # Return the matplotlib figure



# Set up paths
PHOTOS_FOLDER = "/home/appuser/src/pcl_projections"  # Folder containing the 2D projections
image_files = load_image_gallery(PHOTOS_FOLDER)
dataset = ShapeNetSem(
    npoints=2048, 
    split='test', 
    preload=False, 
    use_normals=False,
)
inference_ui = InferenceUI()

current_pcl = None
result_point_cloud = None
with gr.Blocks() as demo:
    gallery = gr.Gallery(image_files)

    # Actions
    with gr.Row():
        point_cloud_button = gr.Button("Run Inference")

    # Output for displaying the selected image path
    selected_image = gr.Textbox(label="Selected Image #", visible=False)

    # Output for showing the inference result
    #inference_result = gr.Textbox(label="Inference Result")
    result_point_cloud_plot = gr.Plot(label="Inference Result (3D Point Cloud)")

    gallery.select(fn = show_warning, inputs = None, outputs = selected_image)

    point_cloud_button.click(
        fn=perform_inference_on_point_cloud,
        inputs=selected_image,
        outputs=result_point_cloud_plot
    )

    

demo.launch()
