import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io
import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision import transforms
from data.data_utils import subsample_instances

root = ''

 # class OfficeHome(Dataset):

#     def __init__(self, split='train', limit = 0, transform=None):
#         self.loader = default_loader
#         self.data = []
#         self.target = []
#         self.target_transform = None
#         self.transform = transform

#         if split == 'train':
#             self.data = f'OfficeHomeDataset_10072016/{split}_2.csv'
#         else:
#             self.data = f'OfficeHomeDataset_10072016/{split}.csv'


#         self.data = pd.read_csv(self.data)

#         self.images = self.data['image'].values
#         self.target = self.data['label'].values 
#         self.uq_idxs = np.array(range(len(self)))

#     def __len__(self):
#         return len(self.target)
    
#     def __getitem__(self, idx):
#         image = self.transform(self.loader(self.images[idx]))
#         target = self.target[idx]

#         if self.target_transform:
#             target = self.target_transform(target)

#         return image, target, self.uq_idxs[idx]
    
class PACS(Dataset):
    def __init__(self, split='train', num = 1, limit = 0, transform=None):
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        

        
        self.data = f'../PACS/{num}/{split}.csv'

        self.data = pd.read_csv(self.data)
        self.images = self.data['image'].values
        self.target = self.data['label'].values 
        self.uq_idxs = np.array(range(len(self)))

        self.pos = self.data['pos'].values
        self.neg = self.data['neg'].values
        self.mask = self.data['mask'].values
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)), 
                transforms.ToTensor()
            ])


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        image = self.transform(self.loader(self.images[idx]))
        pos = self.transform(self.loader(self.pos[idx]))
        neg = self.transform(self.loader(self.neg[idx]))
        mask = torch.load(self.mask[idx], map_location = 'cpu')
        label = self.target[idx]
        if isinstance(image, list):
            masked_image = [im*mask for im in image]
            mask = [mask for _ in range(len(image))]
        else:
            masked_image = image*mask


        return image, masked_image, pos, neg, mask, label, self.uq_idxs[idx]


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

def get_pacs_datasets(train_transform, test_transform, split, train_classes=range(45), prop_train_labels=0.8,
                    split_train_val=False, seed=0):
    np.random.seed(seed)

    train_dataset_labelled = PACS(transform=train_transform, split='train_labelled', num=split)
    train_dataset_unlabelled = PACS(transform=train_transform, split='train_unlabelled', num=split)
    # Split into training and validation sets
    test_dataset = PACS(transform=test_transform, split='test', num=split)

    # Either split train into train and val or use test set as val

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': None,
        'test': test_dataset,
    }

    return all_datasets
    