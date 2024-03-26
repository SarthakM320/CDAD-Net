import os
import pandas as pd
import numpy as np
from copy import deepcopy
import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.data_utils import subsample_instances
from torchvision import transforms
import torch.nn.functional as F
root = ''

class Cifar100(Dataset):
    def __init__(self, split='train', type = 'labelled', limit = 0, transform=None):
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        if split == 'train':
            self.data = pd.read_csv(f'../cifar_100/{split}_{type}.csv')
        else:
            self.data = pd.read_csv(f'../cifar_100/{split}.csv')
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
        return len(self.target)
    
    def __getitem__(self, idx):

        image = self.transform(self.loader('../' + self.images[idx]))
        pos = self.transform(self.loader('../' + self.pos[idx]))
        neg = self.transform(self.loader('../' + self.neg[idx]))
        mask = torch.load('../' + self.mask[idx], map_location = 'cpu')
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(224,224), mode = 'bilinear', align_corners=True).squeeze(0).squeeze(0)
        label = self.target[idx]
        if isinstance(image, list):
            masked_image = [im*mask for im in image]
            mask = [mask for _ in range(len(image))]
        else:
            masked_image = image*mask


        return image, masked_image, pos, neg, mask, label, 0

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = [(p, t,po,ne,ma) for i, (p, t,po,ne,ma) in enumerate(zip(dataset.images,dataset.target, dataset.pos, dataset.neg, dataset.mask)) if i in idxs]
    
    dataset.uq_idxs = list(range(sum(mask)))
    print(len(dataset.uq_idxs))

    return dataset

def subsample_classes(dataset, include_classes=range(5)):

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



def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(5), prop_train_labels=0.8,
                    split_train_val=False, seed=0):
    np.random.seed(seed)

    # whole_training_set = Cifar10(transform=train_transform)


    train_dataset_labelled = Cifar100(transform=train_transform)
    # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    # train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
    # Split into training and validation sets
    # train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    # train_idxs = train_idxs + val_idxs
    # train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)

    # Get unlabelled data
    # unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = Cifar100(transform=train_transform,type='unlabelled')

    # Get test set for all classes
    test_dataset = Cifar100(split='test', transform=test_transform)

    # Either split train into train and val or use test set as val
    # train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': None,
        'test': test_dataset,
    }
    

    return all_datasets
    