import sys
import argparse
from data.load_data import ShapeNetSem
from utils.visualizer import visualize_points
from torch.utils.data import DataLoader

def main(preload):
    # The data can be loaded into split into train, test and validation sets like so 
    # if you have enough memory set the preload flag, it requires about 1.2GB of memory:
    # for the actual training I might implement some kind of caching to save memory (if needed)
    train_dataset = ShapeNetSem(split='train', preload=preload)
    test_dataset = ShapeNetSem(split='test', preload=preload)
    val_dataset = ShapeNetSem(split='val', preload=preload)

    # Create dataloaders for train, test, and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # test the dataloaders and visualize some point clouds
    print('Processing train data')
    train_data = next(iter(train_dataloader))
    pcl, part_label, sn, object_label = train_data
    for i in range(8):
        visualize_points(pcl[i], part_label[i], object_label[i])

    print('Processing test data')
    test_data = next(iter(test_dataloader))
    pcl, part_label, sn, object_label = test_data
    for i in range(8):
        visualize_points(pcl[i], part_label[i], object_label[i])

    print('Processing val data')
    val_data = next(iter(val_dataloader))
    pcl, part_label, sn, object_label = val_data
    for i in range(8):
        visualize_points(pcl[i], part_label[i], object_label[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preload', action='store_true', help='Flag to preload the data')
    args = parser.parse_args()
    if not main(args.preload):
        sys.exit(1)
