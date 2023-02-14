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
from iDataset import iImageNet, iCIFAR100


class NCS(nn.Module):
    def __init__(self, device):
        super(NCS, self).__init__()
        self.alpha = torch.ones(1, dtype=torch.float, device=device, requires_grad=False)
    def forward(self, x):
        return self.alpha * x


def NCS_forward(args, NCSs, output):
    output_NCS = []
    for t in range(len(args.class_num_list)):
        if t==0:
            output_t = output[:, :args.class_num_list[t]]
        else:
            output_t = output[:, args.class_num_list[t-1]:args.class_num_list[t]]
        output_NCS.append(NCSs[t](output_t))
    return torch.cat(output_NCS, dim=1)


def train_model(args, model, old_model, train_loader, optimizer, device, stage, NCSs):
    model.train()

    train_loss = 0.
    train_corrects = [0, 0]
    train_total = 0

    old_class_num = args.class_num_list[stage-1]
    ratio = 1.0 * stage / (stage + 1)
    for images, labels, indexes in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        output,_ = model(images)

        clf_loss = F.cross_entropy(output, labels)

        if stage==0 or args.dis==False:
            loss = clf_loss
        else:
            with torch.no_grad():
                previous_output, _ = old_model(images)
                previous_output = NCS_forward(args, NCSs, previous_output)
            distill_loss = -(1/labels.size(0)) * (F.softmax(previous_output/2, dim=1) * F.log_softmax(output[:, :old_class_num]/2, dim=1)).sum()
            loss = ratio * distill_loss + (1 - ratio) * clf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.restrict:
            for name, param in model.named_parameters():
                if name == 'fc.weight':
                    param.data.clamp_(0)

        for i, topk in enumerate([1, 5]):
            _, pred = output.topk(topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            correct_k = correct.view(-1).float().sum(0, keepdim=True)
            train_corrects[i] += correct_k.item()

        train_total += labels.size(0)
        train_loss += loss.item()

    return train_loss/len(train_loader), train_corrects[0]/train_total, train_corrects[1]/train_total


def test_model(args, model, test_loader, device, stage, NCSs):
    model.eval()

    test_corrects = [0, 0]
    test_total = 0
    with torch.no_grad():
        for images, labels, indexes in tqdm.tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            output,_ = model(images)
            if NCSs != None:
                output = NCS_forward(args, NCSs, output)

            for i, topk in enumerate([1, 5]):
                _, pred = output.topk(topk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                correct_k = correct.view(-1).float().sum(0, keepdim=True)
                test_corrects[i] += correct_k.item()
            
            test_total += labels.size(0)
            
    return test_corrects[0]/test_total, test_corrects[1]/test_total


def learn(args, model, old_model, train_loader, test_loader, device, metric, stage, NCSs):
    weight_decay = args.weight_decay / (stage + 1)
    lr = args.lr #/ (stage + 1)
    logging.info(('lr: %.8f, weight_decay: %.8f'%(lr, weight_decay)))
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    for epoch in range(args.epochs):
        train_loss, train_acc_top1, train_acc_top5 = train_model(args, model, old_model, train_loader, optimizer, device, stage, NCSs)
        logging.info(('Stage: %d, epoch: %d, train loss: %.4f, train acc top1: %.4f, train acc top5: %.4f' % (stage, epoch, train_loss, train_acc_top1, train_acc_top5)))
        scheduler.step()

    test_acc_top1, test_acc_top5 = test_model(args, model, test_loader, device, stage, None)
    logging.info(('Stage: %d, test acc top1: %.4f, test acc top5: %.4f' % (stage, test_acc_top1, test_acc_top5)))
    metric['test_acc'].append([test_acc_top1, test_acc_top5])

    if stage > 0 and args.ncs:
        for ncs in NCSs:
            ncs.alpha.data = torch.ones(1, dtype=torch.float, device=device, requires_grad=False)
   
        w_norm = torch.norm(model.fc.weight.data, p=args.norm, dim=1)
        logging.info((w_norm))
        NCSs[stage].alpha.data = torch.mean(w_norm[:args.class_num_list[stage-1]]) / torch.mean(w_norm[args.class_num_list[stage-1]:args.class_num_list[stage]])
        NCSs[stage].alpha.data.clamp_(max=1.0)

        for ncs in NCSs:
            logging.info((ncs.alpha.data))

        test_acc_top1, test_acc_top5 = test_model(args, model, test_loader, device, stage, NCSs)
        logging.info(('(NCS) Stage: %d, test acc top1: %.4f, test acc top5: %.4f' % (stage, test_acc_top1, test_acc_top5)))
        metric['test_acc'].append([test_acc_top1, test_acc_top5])