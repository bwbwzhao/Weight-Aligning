import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms 
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import copy
from PIL import Image
import os 
import numpy as np 
import pickle
import argparse
import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def creat_logger(log_name):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(asctime)s:  %(levelname)s:  %(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)
    fhlr = logging.FileHandler('./log/%s___%s.log'%(log_name, datetime.now().strftime("%Y%m%d_%H%M%S")))
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    return logger


class iCIFAR100(Dataset):
    def __init__(self, data, targets, transform_type):
        super(iCIFAR100).__init__()
        self.data = data
        self.targets = targets
        if transform_type == 'train':
            self.transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4), 
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                                ])
        elif transform_type == 'test':
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                                ])
        self.target_transform = None

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.targets)


class iImageNet(Dataset):
    def __init__(self, paths, targets, transform_type):
        super(iImageNet).__init__()
        self.paths = paths
        self.targets = targets
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if transform_type == 'train':
            self.transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                                ])
        elif transform_type == 'test':
            self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                                ])
        self.target_transform = None

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path, target = self.paths[index], self.targets[index]
        img = self.pil_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.targets)


def get_IL_data(args):
    # train\test data\targets
    if args.ds == 'cifar':
        train_set_ori = datasets.CIFAR100(root=args.data_path, train=True)
        train_data, train_targets = [img for img in train_set_ori.data], train_set_ori.targets
        test_set_ori = datasets.CIFAR100(root=args.data_path, train=False)
        test_data, test_targets = [img for img in test_set_ori.data], test_set_ori.targets
    elif args.ds == 'imagenet':
        train_data, train_targets, test_data, test_targets = [], [], [], []
        with open(args.data_path + 'train%s.txt'%(args.class_num), "r") as f:
            for line in f.readlines():
                tmp = line.strip().split(' ')
                train_data.append(args.data_path + tmp[0])
                train_targets.append(int(tmp[1]))
        with open(args.data_path + 'val%s.txt'%(args.class_num), "r") as f:
            for line in f.readlines():
                tmp = line.strip().split(' ')
                test_data.append(args.data_path + tmp[0])
                test_targets.append(int(tmp[1]))

    # shuffle class order
    if args.shuffle_order:
        order = list(range(args.class_num))
        np.random.seed(1993)
        np.random.shuffle(order)
        logging.info((order))
        train_targets = list(map(lambda x: order.index(x), train_targets))
        test_targets = list(map(lambda x: order.index(x), test_targets))

    # prepare split data
    train_data_split, train_targets_split, test_data_split, test_targets_split = [], [], [], []
    split_list = [0] + args.class_num_list
    for t in range(1, args.IL_steps + 1):
        train_indices = [i for i, target in enumerate(train_targets) if target in np.arange(split_list[t-1], split_list[t])]
        train_targets_split.append([train_targets[i] for i in train_indices])
        train_data_split.append([train_data[i] for i in train_indices])

        test_indices = [i for i, target in enumerate(test_targets) if target in np.arange(0, split_list[t])]
        test_targets_split.append([test_targets[i] for i in test_indices])
        test_data_split.append([test_data[i] for i in test_indices])

    return train_data_split, train_targets_split, test_data_split, test_targets_split


# get feature 
def replay_id(args, train_data_t, train_targets_t, model, stage, per_class, device):
    if args.random_replay:
        # random
        idx_new_memory = list(range(len(train_targets_t)))
        np.random.shuffle(idx_new_memory)
        idx_matrix = np.zeros((args.task_num_list[stage], per_class), dtype=np.int32)
        counts = np.zeros(args.task_num_list[stage], dtype=np.int32)
        for i in idx_new_memory:
            p = int(train_targets_t[i] % args.task_num_list[stage])
            if counts[p] < per_class:
                idx_matrix[p, counts[p]] = i
                counts[p] += 1
            if counts.sum() == per_class * args.task_num_list[stage]:
                break
    else:
        # herding
        if args.ds == 'imagenet':
            tmp_set = iImageNet(train_data_t, train_targets_t, transform_type='test')
        elif args.ds == 'cifar':
            tmp_set = iCIFAR100(train_data_t, train_targets_t, transform_type='test')
        tmp_loader = torch.utils.data.DataLoader(tmp_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        D_total = []
        model.eval()
        with torch.no_grad():
            for images, labels, indexes in tmp_loader:
                images, labels = images.to(device), labels.to(device)
                _, features = model(images)
                D_total.append(features.cpu().numpy())
        D_total = np.concatenate(D_total)
        D_total = D_total.T
        # new memory id
        idx_matrix = []
        for i in range(args.task_num_list[stage]):
            idx_matrix.append([])
            tmp_indices = [j for j, target in enumerate(train_targets_t) if target == i+(stage*args.classes_perstep)]
            D = D_total[:, tmp_indices]
            D = D / np.linalg.norm(D, axis=0)
            mu  = np.mean(D, axis=1)
            w_t = mu
            step_t = 0
            while not(len(idx_matrix[-1]) == per_class) and step_t<1.5*per_class:
                tmp_t  = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)
                w_t = w_t + mu - D[:, ind_max]
                step_t += 1
                if ind_max not in idx_matrix[-1]:
                    idx_matrix[-1].append(tmp_indices[ind_max])

    idx_matrix = list(np.reshape(np.array(idx_matrix), -1, 'F'))

    return idx_matrix