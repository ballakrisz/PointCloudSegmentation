import torch
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[4]))  # Adds `src` to path
from utils.visualizer import visualize_points

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


class Validator:
    def __init__(self, model, val_loader, cfg, visualize=False):
        self.model = model
        self.val_loader = val_loader
        self.cfg = cfg
        self.visualize = visualize

    def forward(self):
        self.model.eval()
        total_correct = 0
        total_seen = 0
        num_part = self.cfg.MODEL.num_classes
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in OBJECT_PART_LABEL_RANGE.keys()}
        seg_label_to_cat = {label: cat for cat, labels in OBJECT_PART_LABEL_RANGE.items() for label in labels}

        with torch.no_grad():
            for batch_id, data in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                batch_size = data[0].shape[0]
                # extract dcata from data loader
                xyz, features, obj_class, labels = data
                xyz = xyz.cuda()
                features = features.cuda().permute(0,2,1).contiguous()
                labels = labels.cuda()
                obj_class = obj_class.cuda()

                seg_pred = self.model(xyz, features, obj_class)
                #seg_pred = seg_pred.transpose(1, 2)
                seg_pred = torch.argmax(seg_pred, dim=-1)  # Get predicted classes

                # Convert to CPU for evaluation
                seg_pred = seg_pred.cpu().numpy()
                labels = labels.cpu().numpy()

                correct = np.sum(seg_pred == labels)
                total_correct += correct
                total_seen += batch_size * self.cfg.DATA.npoint

                for l in range(num_part):
                    total_seen_class[l] += np.sum(labels == l)
                    total_correct_class[l] += np.sum((seg_pred == l) & (labels == l))

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

                if self.visualize:
                    curr_cat = seg_label_to_cat[labels[0, 0]]
                    curr_acc = correct / (batch_size * self.cfg.DATA.npoint)
                    visualize_points(xyz[0].cpu().numpy(), seg_pred[0], curr_cat, curr_acc, np.max(part_ious), np.min(part_ious), np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            all_shape_ious.extend(shape_ious[cat])
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_iou = np.mean(list(shape_ious.values()))

        test_metrics = {
            "accuracy": total_correct / float(total_seen),
            "class_avg_accuracy": np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=float)),
            "class_avg_iou": mean_shape_iou,
            "instance_avg_iou": np.mean(all_shape_ious),
        }

        return test_metrics

    def __call__(self):
        return self.forward()
