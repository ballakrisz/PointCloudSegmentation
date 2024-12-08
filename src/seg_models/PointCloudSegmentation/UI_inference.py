import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Adds `src` to path
from utils.visualizer import visualize_points
from seg_models.PointCloudSegmentation.openpoints.building_blocks.build import build_model_from_cfg
from seg_models.PointCloudSegmentation.openpoints.utils import EasyConfig, load_checkpoint
from seg_models.PointCloudSegmentation.openpoints.building_blocks.segmentation.pcs import PCS

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

class InferenceUI():
    def __init__(self):
        cfg = EasyConfig()
        cfg.load("/home/appuser/src/seg_models/PointCloudSegmentation/cfg/pcs.yaml", recursive=True)
        self.model = build_model_from_cfg(cfg.MODEL).cuda()
        state_dict = load_checkpoint(self.model, pretrained_path=cfg.pretrained_path)
        self.model.load_state_dict(state_dict['model'])
        self.model.eval()

    def inference(self, data):
        with torch.no_grad():
            batch_size = 1
            xyz = torch.tensor(data[0]).unsqueeze(dim=0).cuda()
            features = torch.tensor(data[1]).unsqueeze(dim=0).cuda().permute(0,2,1).contiguous()
            obj_class = torch.tensor(data[2]).unsqueeze(dim=0).cuda()
            labels = torch.tensor(data[3]).unsqueeze(dim=0).cuda()
            # xyz, features, obj_class, labels = data
            # xyz = xyz.cuda()
            # features = features.cuda().permute(0,2,1).contiguous()
            # labels = labels.cuda()
            # obj_class = obj_class.cuda()

            seg_pred = self.model(xyz, features, obj_class)
            #seg_pred = seg_pred.transpose(1, 2)
            seg_pred = torch.argmax(seg_pred, dim=-1)  # Get predicted classes

            # Convert to CPU for evaluation
            seg_pred = seg_pred.cpu().numpy()
            labels = labels.cpu().numpy()

            correct = np.sum(seg_pred == labels)

            seg_label_to_cat = {label: cat for cat, labels in OBJECT_PART_LABEL_RANGE.items() for label in labels}
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
            curr_cat = seg_label_to_cat[labels[0, 0]]
            curr_acc = correct / (batch_size * 2048)
            return visualize_points(xyz[0].cpu().numpy(), seg_pred[0], curr_cat, curr_acc, np.max(part_ious), np.min(part_ious), np.mean(part_ious))