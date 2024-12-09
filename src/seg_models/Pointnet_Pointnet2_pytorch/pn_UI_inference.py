import torch
import numpy as np
import sys
from pathlib import Path
import importlib
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adds `src` to the system path for easy imports
from utils.visualizer import visualize_points, visualize_points_plotly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory of the script
ROOT_DIR = BASE_DIR  # Root directory for the project
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # Adds the 'models' folder to the path to access model definitions

# Define segmentation categories and their corresponding labels
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

# Create a mapping of segmentation labels to categories
seg_label_to_cat = {}  
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

# Load class names from a file and map them to integers
cat = {}
with open("/home/appuser/shapenetcore_partanno_segmentation_benchmark_v0_normal/synsetoffset2category.txt", 'r') as f:
    for line in f:
        ls = line.strip().split()
        cat[ls[0]] = ls[1]
cat = {k: v for k, v in cat.items()}
classes_original = dict(zip(cat, range(len(cat))))

# Reverse the class dictionary for easier lookup
classes = {}
for i in cat.keys():
    classes[i] = classes_original[i]
classes = {v: k for k, v in classes.items()}

# Function to convert class labels into one-hot encoded format
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

# Define the PnInferenceUI class that performs inference with the PointNet++ model
class PnInferenceUI:
    def __init__(self) -> None:
        model_name = "pointnet2_part_seg_msg"  # Model name
        MODEL = importlib.import_module(model_name)  # Dynamically import model
        self.model = MODEL.get_model(50, True).cuda()  # Load model with 50 classes, move to GPU
        checkpoint_folder = "/home/appuser/checkpoints/PointNet2PartSeg/"  # Directory for saved checkpoints
        checkpoints = [torch.load(os.path.join(checkpoint_folder, f)) for f in os.listdir(checkpoint_folder) if f.endswith('.pth')]
        checkpoint = max(checkpoints, key=lambda x: x['test_acc'])  # Get the checkpoint with the highest test accuracy
        
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        self.model.eval()  # Set the model to evaluation mode

    def inference(self, data):
        """ Perform inference using the loaded model and input data """
        with torch.no_grad():  # Disable gradient calculation
            test_metrics = {}  # Dictionary to store evaluation metrics
            total_correct = 0  # Total number of correct predictions
            total_seen = 0  # Total number of points seen
            total_seen_class = [0 for _ in range(50)]  # Total seen per class
            total_correct_class = [0 for _ in range(50)]  # Total correct per class
            shape_ious = {cat: [] for cat in seg_classes.keys()}  # IOU scores for each shape category
            seg_label_to_cat = {}  # Re-map labels to categories

            # Map each label to its corresponding category
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            points = torch.tensor(data[0]).unsqueeze(dim=0).cuda()  # Load input point cloud data
            cur_batch_size, NUM_POINT, _ = points.size()  # Get batch size and number of points
            label = torch.tensor(data[1]).unsqueeze(dim=0).long().cuda()  # Load ground truth labels
            target = torch.tensor(data[2]).unsqueeze(dim=0).long().cuda()  # Load target segmentation labels
            points = points.transpose(2, 1)  # Transpose to match model input format
            vote_pool = torch.zeros(target.size()[0], target.size()[1], 50).cuda()  # Initialize vote pool for predictions

            pcl = points  # Store point cloud data
            object_class = classes[int(label.squeeze().cpu().numpy())]  # Get the object class from the label

            # Move all tensors to GPU
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            vote_pool = torch.zeros(target.size()[0], target.size()[1], 50).cuda()  # Reset vote pool

            # Perform inference with model multiple times (voting mechanism)
            for _ in range(3):
                seg_pred, _ = self.model(points, to_categorical(label, 16))  # Model prediction
                lables = np.argmax(seg_pred.cpu().data.numpy()[0], 1)  # Get predicted labels
                vote_pool += seg_pred  # Add to vote pool

            seg_pred = vote_pool / 3  # Average predictions from 3 runs
            cur_pred_val = seg_pred.cpu().data.numpy()  # Convert predictions to numpy
            cur_pred_val_logits = cur_pred_val  # Store logits for later use
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)  # Initialize final predictions
            target = target.cpu().data.numpy()  # Convert target to numpy

            # Process predictions and compute accuracy metrics
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]  # Get category of the current object
                logits = cur_pred_val_logits[i, :, :]  # Get logits for the current batch
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]  # Get final predicted labels

            correct = np.sum(cur_pred_val == target)  # Count number of correct predictions
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)  # Update total seen points

            # Calculate per-class accuracy
            for l in range(50):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            # Calculate per-category IOU
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]  # Initialize IOU scores for parts
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # If part is not present, IOU is 1
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))  # Calculate IOU for the part
                shape_ious[cat].append(np.mean(part_ious))  # Store average IOU for the shape

            # Calculate the best, worst, and average IOU for the shape
            worst_part_iou = np.min(part_ious)
            best_part_iou = np.max(part_ious)
            avg_part_iou = np.mean(part_ious)
            acc = correct / (cur_batch_size * NUM_POINT)  # Calculate overall accuracy
            pcl = pcl.transpose(2, 1)  # Transpose point cloud data back
            return visualize_points_plotly(pcl[0, :, :3].cpu().numpy(), cur_pred_val[0], object_class, acc, best_part_iou, worst_part_iou, avg_part_iou)
