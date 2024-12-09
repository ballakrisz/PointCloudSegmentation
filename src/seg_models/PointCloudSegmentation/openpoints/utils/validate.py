import torch
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))  # Adds `src` to path
from utils.visualizer import visualize_points
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

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


class_to_label = {
    'Airplane': 0,
    'Bag': 1,
    'Cap': 2,
    'Car': 3,
    'Chair': 4,
    'Earphone': 5,
    'Guitar': 6,
    'Knife': 7,
    'Lamp': 8,
    'Laptop': 9,
    'Motorbike': 10,
    'Mug': 11,
    'Pistol': 12,
    'Rocket': 13,
    'Skateboard': 14,
    'Table': 15
}


class Validator:
    """
    Class for running validation both during training and testing
    """
    def __init__(self, model, val_loader, cfg, visualize=False):
        """
        Args:
            model: Model to validate
            val_loader: DataLoader for validation
            cfg: Config object
            visualize: Whether to visualize the predictions
        """
        self.model = model
        self.val_loader = val_loader
        self.cfg = cfg
        self.visualize = visualize

    def forward(self):
        # Setup model for evaluation and some variables
        self.model.eval()
        total_correct = 0
        total_seen = 0
        num_part = self.cfg.MODEL.num_classes
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in OBJECT_PART_LABEL_RANGE.keys()}
        seg_label_to_cat = {label: cat for cat, labels in OBJECT_PART_LABEL_RANGE.items() for label in labels}

        # confusion matrces (If you'd like to see them remove the comments for later rows (all of them))
        class_wise_conf_matrix = np.zeros((16, 16))
        plane_conf_matrix = np.zeros((5, 5))

        # don't track gradients (no point)
        with torch.no_grad():
            # Process one epoch
            for batch_id, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                batch_size = data[0].shape[0]
                # extract dcata from data loader
                xyz, features, obj_class, labels = data
                xyz = xyz.cuda()
                features = features.cuda().permute(0,2,1).contiguous()
                labels = labels.cuda()
                obj_class = obj_class.cuda()

                # Inference and get the predicted classes by taking the argmax
                seg_pred = self.model(xyz, features, obj_class)
                seg_pred = torch.argmax(seg_pred, dim=-1)  # Get predicted classes

                # Convert to CPU for evaluation
                seg_pred = seg_pred.cpu().numpy()
                labels = labels.cpu().numpy()

                # Calculate correct and total seen scenarios
                correct = np.sum(seg_pred == labels)
                total_correct += correct
                total_seen += batch_size * self.cfg.DATA.npoint

                # Calculate class-wise accuracy
                for l in range(num_part):
                    total_seen_class[l] += np.sum(labels == l)
                    total_correct_class[l] += np.sum((seg_pred == l) & (labels == l))

                # Calculate shape-wise IoU
                for i in range(batch_size):
                    segp = seg_pred[i]
                    segl = labels[i]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = []
                    for l in OBJECT_PART_LABEL_RANGE[cat]:
                        if np.sum(segl == l) == 0 and np.sum(segp == l) == 0:
                            part_ious.append(1.0)
                        else:
                            iou = np.sum((segl == l) & (segp == l)) / np.sum((segl == l) | (segp == l))
                            part_ious.append(iou)
                    shape_ious[cat].append(np.mean(part_ious))

                # for i in range(batch_size):
                #     obj_cls = obj_class[i].item()
                #     if obj_cls == 0:
                #         for j in range(self.cfg.DATA.npoint):
                #             part_true = labels[i, j]
                #             part_pred = seg_pred[i, j]
                #             if (part_pred > 3):
                #                 part_pred = 4
                #             plane_conf_matrix[part_true, part_pred] += 1

                # for i in range(batch_size):
                #     segp = seg_pred[i]  # Predicted labels for the current point cloud
                #     segl = labels[i]    # Ground truth labels for the current point cloud
                    
                #     # Initialize a counter for predicted classes
                #     predicted_class_counts = Counter()
                    
                #     # Loop through the predicted parts (segp) and ground truth parts (segl)
                #     for point_idx in range(len(segl)):
                #         predicted_class = segp[point_idx]   # Predicted label for the point
                #         true_class = segl[point_idx]       # True label for the point
                        
                #         # Count how many times each predicted class occurs
                #         predicted_class_counts[predicted_class] += 1
                    
                #     # Find the predicted class with the majority of points (highest count)
                #     predicted_class = predicted_class_counts.most_common(1)[0][0]
                    
                #     # Map the predicted class to the object category
                #     obj_pred = class_to_label[seg_label_to_cat[predicted_class]]  # Map the first label in the object to its category
                #     class_wise_conf_matrix[obj_class[i].item(), obj_pred] += 1

                # Visualize the predictions (though this is deprecated, please use the gradio visualization instead)
                if self.visualize:
                    curr_cat = seg_label_to_cat[labels[0, 0]]
                    curr_acc = correct / (batch_size * self.cfg.DATA.npoint)
                    visualize_points(xyz[0].cpu().numpy(), seg_pred[0], curr_cat, curr_acc, np.max(part_ious), np.min(part_ious), np.mean(part_ious))


        # plt.figure(figsize=(20, 12))
        # sns.heatmap(plane_conf_matrix, annot=True, cmap='Blues', 
        #     xticklabels=['0', '1', '2', '3', 'other class'], yticklabels=['0', '1', '2', '3', 'other class'],
        #     cbar=True, annot_kws={"size": 16})
        # plt.title('Confusion Matrix')
        # plt.xlabel('Predicted Part Labels')
        # plt.ylabel('True Part Labels')
        # plt.show()

        # Calculate class-wise and instance-wise IoU-s
        all_shape_ious = []
        for cat in shape_ious.keys():
            all_shape_ious.extend(shape_ious[cat])
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_iou = np.mean(list(shape_ious.values()))

        # Construct the results dictionary
        test_metrics = {
            "accuracy": total_correct / float(total_seen),
            "class_avg_accuracy": np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float)),
            "class_avg_iou": mean_shape_iou,
            "instance_avg_iou": np.mean(all_shape_ious),
        }

        return test_metrics

    def __call__(self):
        return self.forward()
