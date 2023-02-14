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
import math
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from preresnet18 import preresnet18
from preresnet32 import preresnet32, NormLinear
from iDataset import iImageNet, iCIFAR100
from iDataset import get_IL_data, creat_logger, replay_id
from iLearn import NCS, NCS_forward, train_model, test_model, learn


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--ds', default='imagenet', type=str)
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--milestones', type=int, nargs='+', default=[30,60,80,90])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--IL_steps', default=10, type=int)
    parser.add_argument('--data_path', default='/cache/imagenet/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--norm', default=1, type=int)
    parser.add_argument('--shuffle_order', dest='shuffle_order', action='store_false')
    parser.add_argument('--withbias', dest='withbias', action='store_true')
    parser.add_argument('--random_replay', dest='random_replay', action='store_true')
    parser.add_argument('--restrict', dest='restrict', action='store_false')
    parser.add_argument('--dis', dest='dis', action='store_false')
    parser.add_argument('--ncs', dest='ncs', action='store_false')
    parser.add_argument('--savedir', default='ours', type=str)
    args, unparsed = parser.parse_known_args()
    args_dict = vars(args)
    args_dict['memory_size'] = args.class_num * 20
    args_dict['classes_perstep'] = int(args.class_num / args.IL_steps)
    args_dict['class_num_list'] = [args.classes_perstep*(i+1) for i in range(args.IL_steps)]
    args_dict['task_num_list'] = [args.classes_perstep for i in range(args.IL_steps)]

    logger = creat_logger('%s_%s_%s'%(args.ds, args.class_num, args.IL_steps))
    logger.info((args))

    device = torch.device('cuda:%d'%(args.device))

    if args.ds=='imagenet':
        net = preresnet18
        idataset = iImageNet
    elif args.ds=='cifar':
        net = preresnet32
        idataset = iCIFAR100

    # prepare data and targets
    train_data_split, train_targets_split, test_data_split, test_targets_split = get_IL_data(args)

    # prepare model
    model = net()
    model.fc = nn.Linear(model.fc.in_features, args.class_num_list[0], bias=args.withbias)
    # model.load_state_dict(torch.load('./trained_models/tmp/imagenet1000_10/imagenet1000_10_0.pth', map_location=device))
    model.to(device)
    old_model = None

    # memory set init
    memory_data, memory_targets= [], []
    # build bias layers
    NCSs = []
    for _ in range(args.IL_steps):
        NCSs.append(NCS(device))
    # metric init
    metric = {'test_acc':[]}

    # incremental learning process
    for t in range(args.IL_steps):
        logger.info(('========================================== %d ===============================================' %(t)))

        # update model layer
        if t != 0:
            in_features, out_features = model.fc.in_features, model.fc.out_features
            fc_weight = copy.deepcopy(model.fc.weight.data)
            new_fc = nn.Linear(in_features, args.class_num_list[t], bias=args.withbias).to(device)
            #new_fc.weight.data[:out_features] = fc_weight
            del model.fc
            model.fc = new_fc

        # prepare train dataset for this IL step
        train_data_t, train_targets_t = train_data_split[t], train_targets_split[t]
        train_data_old, train_targets_old = sum(memory_data, []), sum(memory_targets, []) 
        logger.info(('old:', len(train_data_old)))
        train_data_t_cat, train_targets_t_cat = train_data_t + train_data_old, train_targets_t + train_targets_old 
        train_set_t = idataset(train_data_t_cat, train_targets_t_cat, transform_type='train')
        logger.info(('train_set_t', len(train_set_t)))
        # prepare test dataset for this IL step
        test_data_t, test_targets_t = test_data_split[t], test_targets_split[t]
        test_set_t = idataset(test_data_t, test_targets_t, transform_type='test')
        logger.info(('test_set_t', len(test_set_t)))
        # prepare datalader for this IL step
        train_loader_t = torch.utils.data.DataLoader(train_set_t, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader_t = torch.utils.data.DataLoader(test_set_t, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # training for this IL step
        if t == 0:
            print(test_model(args, model, test_loader_t, device, t, NCSs))
        if t != 0:
            learn(args, model, old_model, train_loader_t, test_loader_t, device, metric, t, NCSs)
            s_dir = './trained_models/%s_%s%d_%d/'%(args.savedir, args.ds, args.class_num, args.IL_steps)
            if not os.path.exists(s_dir):
                os.makedirs(s_dir)
            torch.save(model.state_dict(), s_dir + '%s%d_%d_%d.pth'%(args.ds, args.class_num, args.IL_steps, t))

        # all train finished
        if t == args.IL_steps - 1:
            break

        # save old model
        old_model = copy.deepcopy(model)
        for param in old_model.parameters():
            param.requires_grad = False
        old_model.eval()

        # shorten old memory
        per_class = args.memory_size // args.class_num_list[t]
        for i in range(t):
            memory_data[i] = memory_data[i][:per_class * args.task_num_list[i]]
            memory_targets[i] = memory_targets[i][:per_class * args.task_num_list[i]]
            logger.info(('shorten, memory_data[%d], memory_targets[%d], %d, %d'%(i, i, len(memory_data[i]), len(memory_targets[i]))))
        # add new memory
        idx_matrix = replay_id(args, train_data_t, train_targets_t, model, t, per_class, device)
        memory_data.append([train_data_t[i] for i in idx_matrix])
        memory_targets.append([train_targets_t[i] for i in idx_matrix])

    # test acc summary
    for test_acc_t in metric['test_acc']:
        logger.info((test_acc_t))