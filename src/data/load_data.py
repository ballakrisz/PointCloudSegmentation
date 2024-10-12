import kagglehub
from torch.utils.data import Dataset
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# from pympler import asizeof

SYNSET_DICT = {
    '02691156': 'Airplane',
    '02773838': 'Bag',
    '02954340': 'Cap',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03261776': 'Earphone',
    '03467517': 'Guitar',
    '03624134': 'Knife',
    '03636649': 'Lamp',
    '03642806': 'Laptop',
    '03790512': 'Motorbike',
    '03797390': 'Mug',
    '03948459': 'Pistol',
    '04099429': 'Rocket',
    '04225987': 'Skateboard',
    '04379243': 'Table'
}

# normalize the data, so it has a mean of 0, and is within the range [-1, 1] (unit circle)
def normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ShapeNetSem(Dataset):
    def __init__(self,npoints = 2500, split = 'test', preload=False) -> None:
        """
        Initialize the LoadData class.

        Args:
            split (str): The split type for the data.
            preload (bool, optional): Whether to preload the data. Defaults to False.
        """
        super().__init__()
        self.split = split
        self.preload = preload
        self.npoints = npoints
        
        # Download the data (if not already downloaded) --> caching is handled by kagglehub
        self.download_path = kagglehub.dataset_download("mitkir/shapenet") + "/shapenetcore_partanno_segmentation_benchmark_v0_normal/"

        # config files for the test, train and val splits
        self.test_split = self.download_path + "train_test_split/shuffled_test_file_list.json"
        self.train_split = self.download_path + "train_test_split/shuffled_train_file_list.json"
        self.val_split = self.download_path + "train_test_split/shuffled_val_file_list.json"
        
        # Dictionary to store the data
        self.data_dict = {}
        
        # Load the data (if preload is False, only the filenames will be stored)
        self.prepare_data()
        
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        if self.preload:
            key = list(self.data_dict.keys())[index]
            data = self.data_dict[key]
        else:
            key = list(self.data_dict.keys())[index]
            path = self.data_dict[key]['path']
            npz_data = np.load(path)

            # Store the data in a dictionary (it might be unncessary .. TODO)
            data = {
                'part_label': npz_data['part_label'],
                'surface_normal': npz_data['sn'],
                'point_cloud': npz_data['pc'],
                'object_label': SYNSET_DICT[key.split('/')[1]]
            }
            npz_data.close()
        
        # Randomly sample npoints from the point cloud --> same techinque as in the PointNet++ paper
        choice = np.random.choice(len(data['point_cloud']), self.npoints, replace=True)
        data['point_cloud'] = normalize(data['point_cloud'])[choice, :]
        data['part_label'] = data['part_label'][choice]
        data['surface_normal'] = data['surface_normal'][choice, :]

        return data['point_cloud'], data['part_label'], data['surface_normal'], data['object_label']
        
    def prepare_data(self):
        if self.split == 'train':
            split=self.train_split
        elif self.split == 'test':
            split=self.test_split
        elif self.split == 'val':
            split=self.val_split
        else:
            raise ValueError('Invalid split')

        with open(split, 'r') as file:
            files = json.load(file)
            for file in files:
                components = file.split('/')
                new_components = components[1:]
                path = '/'.join(new_components)
                
                # if not preload, only store the filepaths
                if not self.preload:
                    self.data_dict[file] = {
                        'path': self.download_path + path + "_8x8.npz"
                    }
                    continue
                
                # preload the data (this is the faster option, if it fits in memory)
                data = np.load(self.download_path + path + "_8x8.npz")

                # Store the data in the dictionary with filename as the key
                self.data_dict[file] = {
                    'part_label': data['part_label'],
                    'surface_normal': data['sn'],
                    'point_cloud': data['pc'],
                    'object_label': SYNSET_DICT[path.split('/')[0]]
                }
                data.close()

# mb_size = 0
# for dataset in [train_dataset, test_dataset, val_dataset]:
#     size_in_bytes = asizeof.asizeof(dataset.data_dict)
#     mb_size += size_in_bytes / (1024 * 1024)

# print(f"Size of dataset: {mb_size} MB")

        
    