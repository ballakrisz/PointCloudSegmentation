import torch
import numpy as np
import sys
from pathlib import Path
import importlib
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adds `src` to path
from utils.visualizer import visualize_points, visualize_points_plotly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

cat = {}
with open("/home/appuser/shapenetcore_partanno_segmentation_benchmark_v0_normal/synsetoffset2category.txt", 'r') as f:
    for line in f:
        ls = line.strip().split()
        cat[ls[0]] = ls[1]
cat = {k: v for k, v in cat.items()}
classes_original = dict(zip(cat, range(len(cat))))

classes = {}
for i in cat.keys():
    classes[i] = classes_original[i]
classes = {v: k for k, v in classes.items()}

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

class PnInferenceUI:
    def __init__(self) -> None:
        model_name = "pointnet2_part_seg_msg"
        MODEL = importlib.import_module(model_name)
        self.model = MODEL.get_model(50, True).cuda()
        checkpoint_folder = "/home/appuser/checkpoints/PointNet2PartSeg/"
        checkpoints = [torch.load(os.path.join(checkpoint_folder, f)) for f in os.listdir(checkpoint_folder) if f.endswith('.pth')]
        checkpoint = max(checkpoints, key=lambda x: x['test_acc'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def inference(self, data):
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(50)]
            total_correct_class = [0 for _ in range(50)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            points = torch.tensor(data[0]).unsqueeze(dim=0).cuda()
            cur_batch_size, NUM_POINT, _ = points.size()
            label = torch.tensor(data[1]).unsqueeze(dim=0).long().cuda()
            target = torch.tensor(data[2]).unsqueeze(dim=0).long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], 50).cuda()

            pcl = points
            object_class = classes[int(label.squeeze().cpu().numpy())]

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            vote_pool = torch.zeros(target.size()[0], target.size()[1], 50).cuda()


            for _ in range(3):
                seg_pred, _ = self.model(points, to_categorical(label, 16))
                lables = np.argmax(seg_pred.cpu().data.numpy()[0],1)
                vote_pool += seg_pred

            seg_pred = vote_pool / 3
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(50):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

            worst_part_iou = np.min(part_ious)
            best_part_iou = np.max(part_ious)
            avg_part_iou = np.mean(part_ious)
            acc = correct / (cur_batch_size * NUM_POINT)
            pcl = pcl.transpose(2, 1)
            return(visualize_points_plotly(pcl[0,:,:3].cpu().numpy(), cur_pred_val[0], object_class, acc, best_part_iou, worst_part_iou, avg_part_iou))

