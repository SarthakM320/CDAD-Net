import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances

root = ''

class OfficeHome(Dataset):

    def __init__(self, split='train', limit = 0, transform=None):
        self.loader = default_loader
        self.data = []
        self.target = []
        self.target_transform = None
        self.transform = transform

        if split == 'train':
            self.data = f'OfficeHomeDataset_10072016/{split}_2.csv'
        else:
            self.data = f'OfficeHomeDataset_10072016/{split}.csv'


        self.data = pd.read_csv(self.data)

        self.images = self.data['image'].values
        self.target = self.data['label'].values 
        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        image = self.transform(self.loader(self.images[idx]))
        target = self.target[idx]

        if self.target_transform:
            target = self.target_transform(target)

        return image, target, self.uq_idxs[idx]
    

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = [(p, t) for i, (p, t) in enumerate(zip(dataset.images,dataset.target)) if i in idxs]
    print(len(dataset.uq_idxs))
    print(len(mask))
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset

def subsample_classes(dataset, include_classes=range(45)):

    cls_idxs = [i for i, (p, t) in enumerate(zip(dataset.images,dataset.target)) if t in include_classes]

    # TODO: Don't transform targets for now
    # target_xform_dict = {}
    # for i, k in enumerate(include_classes):
    #     target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    all_targets = [t for i, (p, t) in enumerate(zip(train_dataset.images,train_dataset.target))]
    train_classes = np.unique(all_targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(all_targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_officehome_datasets(train_transform, test_transform, train_classes=range(45), prop_train_labels=0.8,
                    split_train_val=False, seed=0):
    np.random.seed(seed)

    whole_training_set = OfficeHome(transform=train_transform, split='train')
    train_dataset_labelled = deepcopy(whole_training_set)

    # Split into training and validation sets
    val_dataset_labelled = OfficeHome(transform=train_transform,split = 'val')
    val_dataset_labelled.transform = test_transform
    test_dataset = OfficeHome(transform=test_transform, split='test')

    # Either split train into train and val or use test set as val

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': None,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets
    