import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # Adds `src` to path
from utils.visualizer import visualize_points
from data.load_data import ShapeNetSem

SAVE_PATH = "/home/appuser/pcl_projections/"

def rotation_matrix(yaw, pitch, roll):
    """
    Create a rotation matrix for the given yaw, pitch, and roll angles (in degrees).
    """
    # Convert angles from degrees to radians
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Rotation matrices around each axis
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    # Combined rotation matrix
    return R_yaw @ R_pitch @ R_roll

def draw_circle(image, center, radius, intensity=1):
    """
    Draw a filled circle on a 2D numpy array (image).

    Args:
        image: 2D numpy array representing the image.
        center: (x, y) tuple for the circle's center.
        radius: Radius of the circle.
        intensity: Intensity value to fill the circle.
    """
    cx, cy = center
    for x in range(max(0, cx - radius), min(image.shape[1], cx + radius + 1)):
        for y in range(max(0, cy - radius), min(image.shape[0], cy + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                image[y, x] = intensity

def project_point_cloud(point_cloud, yaw, pitch, roll, H, W, circle_radius=5):
    """
    Project a 3D point cloud onto a 2D image plane after rotating it.
    
    Args:
        point_cloud: Nx3 numpy array with (x, y, z) coordinates.
        yaw, pitch, roll: Rotation angles in degrees.
        H, W: Height and width of the output image.
    
    Returns:
        2D numpy array representing the image.
    """
    # Step 1: Rotate the point cloud
    R = rotation_matrix(yaw, pitch, roll)
    rotated_points = point_cloud @ R.T

    # Step 2: Project onto the XY plane (ignore z)
    x, y = rotated_points[:, 0], rotated_points[:, 1]

    # Step 3: Normalize to fit within image bounds
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # Calculate the aspect ratio
    aspect_ratio = x_range / y_range

    # Adjust image size based on aspect ratio
    if aspect_ratio > 1:
        # If the x range is larger, adjust W
        W = int(H * aspect_ratio)
    else:
        # If the y range is larger, adjust H
        H = int(W / aspect_ratio)

    # Normalize to fit within adjusted image bounds
    u = ((x - x.min()) / x_range * (W - 1)).astype(int)
    v = ((y - y.min()) / y_range * (H - 1)).astype(int)

    # Step 4: Create the image and plot points
    yellow = (0.7, 0.7, 0.5)  # Light yellow color in RGB
    image = np.ones((H, W, 3))  # White background
    for i in range(len(u)):
        if 0 <= u[i] < W and 0 <= v[i] < H:  # Ensure points fall within bounds
            draw_circle(image, (u[i], v[i]), circle_radius, intensity=yellow)

    return image

# Example usage
dataset = ShapeNetSem(
    npoints=2048, 
    split='test', 
    preload=False, 
    use_normals=False,
)

for i in range(len(dataset)):
    data = dataset[i]
    xyz, features, obj_class, labels = data
    point_cloud = xyz

    # Extract x and y coordinates (ignore z)
    x, y = point_cloud[:, 0], point_cloud[:, 1]

    # Image resolution (will be adjusted based on aspect ratio)
    H, W = 512, 512
    yaw, pitch, roll = 0, 0, 0

    projected_image = project_point_cloud(point_cloud, yaw, pitch, roll, H, W)

    # Optional: Enlarge points for better visibility
    plt.figure(figsize=(6, 6))
    plt.imshow(projected_image, cmap='gray', origin='lower')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axis('off')
    plt.savefig(f"{SAVE_PATH}/point_cloud_{i}.png")
    #plt.show()
