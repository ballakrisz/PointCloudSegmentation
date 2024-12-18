import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np 
import sys
import plotly.graph_objects as go

OBJECT_PART_LABEL_RANGE = {
    'Airplane': list(range(4)),
    'Bag': list(range(4, 6)),
    'Cap': list(range(6, 8)),
    'Car': list(range(8, 12)),
    'Chair': list(range(12, 16)),
    'Earphone': list(range(16, 19)),
    'Guitar': list(range(19, 22)),
    'Knife': list(range(22, 24)),
    'Lamp': list(range(24, 28)),
    'Laptop': list(range(28, 30)),
    'Motorbike': list(range(30, 36)),
    'Mug': list(range(36, 38)),
    'Pistol': list(range(38, 41)),
    'Rocket': list(range(41, 44)),
    'Skateboard': list(range(44, 47)),
    'Table': list(range(47, 50))
}

PART_LABEL_COLORS = {
    0: 'purple',
    1: 'tan',
    2: 'blue',
    3: 'green',
    4: 'pink',
    5: 'red'
}

def on_key(event):
    if event.key == 'escape':
        sys.exit(0)
    elif event.key == 'right':
        plt.close()

def visualize_points_plotly(point_cloud, part_label, object_label, acc, best_part_iou, worst_part_iou, avg_part_iou, pointnet=False):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Create a DataFrame from point cloud and part labels
    ss = np.array([x, y, z, part_label]).transpose(1, 0)
    df = pd.DataFrame(ss, columns=['x', 'y', 'z', 'part_label'])
    grouped = df.groupby('part_label')
    grouped_data = {category: group[['x', 'y', 'z']].values for category, group in grouped}
    
    # Create Plotly figure
    fig = go.Figure()

    # Plot each group with its corresponding color
    for category, group in grouped_data.items():
        label_range = OBJECT_PART_LABEL_RANGE[object_label]
        idx = max(label_range) - category 
        
        # If the part labels are not part of the object, set it to red
        if idx not in PART_LABEL_COLORS:
            idx = 5  # Default to red or another color

        fig.add_trace(go.Scatter3d(
            x=group[:, 0],
            y=group[:, 1],
            z=group[:, 2],
            mode='markers',
            marker=dict(size=2, color=PART_LABEL_COLORS[idx], opacity=0.8),
            name=f'Part {category}'
        ))

    # Set axis labels and limits
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(
                title='X Label',
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False
            ),
            yaxis=dict(
                title='Y Label',
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False
            ),
            zaxis=dict(
                title='Z Label',
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False
            ),
        ),
        title=f'{object_label}<br>Accuracy: {acc}<br>Best part iou: {best_part_iou}<br>Worst part iou: {worst_part_iou}<br>Avg part iou: {avg_part_iou}',
        title_x=0.5,  # Center title horizontally
        title_y=0.95,  # Position title vertically
        title_font=dict(size=18, color='black'),  # Title font size and color
        showlegend=False
    )

    # Make sure the plot is interactive
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    
    return fig


def visualize_points(point_cloud, part_label, object_label, acc, best_part_iou, worst_part_iou, avg_part_iou, pointnet=False):
    """
    @DEPRECATED
    Plotply is better suited for 3D visualizations and gradio can display it as an actual 3D scene
    """
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # create fig
    fig = plt.figure(figsize=(25,12.5))
    ax = fig.add_subplot(111, projection='3d')
    
    # split the data into groups based on the part label
    ss = np.array([x,y,z,part_label]).transpose(1,0)
    df = pd.DataFrame(ss, columns=['x', 'y', 'z', 'part_label'])
    grouped = df.groupby('part_label')
    grouped_data = {category: group[['x', 'y', 'z']].values for category, group in grouped}
    
    # plot the groups with their corresponding colors
    for category, group in grouped_data.items():
        object_label = object_label
        label_range = OBJECT_PART_LABEL_RANGE[object_label]
        idx = max(label_range) - category 
        # If the part labels is not part of the object, set it to red
        if idx not in PART_LABEL_COLORS:
            idx = 5
        ax.scatter(group[:,0], group[:,1], group[:,2], c=PART_LABEL_COLORS[idx], marker='o')

    # Set labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set axis limits so the point cloud is not distorted
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    plt.title(f'{object_label}\n Accuracy: {acc}\n Best part iou: {best_part_iou}\n Worst part iou: {worst_part_iou}\n Avg part iou: {avg_part_iou}')
    plt.axis('off')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()