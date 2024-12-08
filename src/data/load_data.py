from torch.utils.data import Dataset
import json
import numpy as np
import torch
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
    def __init__(self,npoints = 2500, split = 'test', preload=False, use_normals=False, transforms = None, pointNet=False) -> None:
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
        self.use_normals = use_normals
        self.transforms = transforms
        self.pointNet = pointNet
        self.catfile = "/home/appuser/shapenetcore_partanno_segmentation_benchmark_v0_normal/synsetoffset2category.txt"
        self.num_classes = 16
        self.cat = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        
        # Download the data (if not already downloaded) --> caching is handled by kagglehub
        self.download_path = "/home/appuser/shapenetcore_partanno_segmentation_benchmark_v0_normal/"

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
        
        # if using normals, concatenate it to the point cloud
        features = None
        if self.use_normals:
            pcl = np.concatenate((data['point_cloud'], data['surface_normal']), axis=1)
        else:
            pcl = data['point_cloud']
            features = data['surface_normal']

        # normalize the point cloud
        pcl[:, 0:3] = normalize(pcl[:, 0:3])
        
        # Randomly sample npoints from the point cloud --> same techinque as in the PointNet++ paper
        choice = np.random.choice(len(pcl), self.npoints, replace=True)
        pcl = pcl[choice, :]
        if features is not None:
            features = features[choice, :]
        point_labels = data['part_label'][choice]
        point_labels = np.array([point_labels]).astype(np.int32).squeeze()

        # extract the object label for class-wise performrance analysis
        object_class = data['object_label']
        object_label = np.array([self.classes[object_class]]).astype(np.int64)


        if self.transforms:
            pcl = torch.from_numpy(pcl).float()
            for transform in self.transforms:
                pcl = transform(pcl)

        if self.pointNet:
            return pcl, object_label, point_labels

        return pcl, features, object_label, point_labels
        
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

        
    