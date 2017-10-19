import math
import torch
import shutil
import time
import os
import random
from PIL import Image
from easydict import EasyDict as edict
import yaml
import torch.utils.data as data
import numpy as np

# 

#
def read_data_file_with_dict(filename, root=None):

    lists = []
    dicts = {}
    cnt = 0
    with open(filename, 'r') as fp:
        line = fp.readline()
        while line:
            info = line.strip().split(' ')
            if root is not None:
                #info[0] = root + info[0]
                info[0] = os.path.join(root, info[0])
            item = (info[0], int(info[1]))
            lists.append(item)

            if int(info[1]) not in dicts:
                dicts[int(info[1])] = []
            dicts[int(info[1])].append(cnt)
            cnt += 1

            line = fp.readline()

    max_number = 0
    for x in dicts:
        max_number = max(max_number, len(dicts[x]))
    return lists, dicts, max_number

def read_data_file(filename, root=None):

    lists = []
    with open(filename, 'r') as fp:
        line = fp.readline()
        while line:
            info = line.strip().split(' ')
            if root is not None:
                #info[0] = root + info[0]
                info[0] = os.path.join(root, info[0])
            item = (info[0], int(info[1]))
            lists.append(item)
            line = fp.readline()

    return lists

def pil_loader(path):

    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def label_shuffle(dicts, max_number, idx):

    index = [i for i in range(max_number)]
    cnt = 0

    for x in dicts:
        info = dicts[x]
        print info
        random.shuffle(info)
        length = len(info)
        random.shuffle(index)
        for y in index:
            ids = y % length
            idx[cnt] = info[ids]
            cnt += 1

class MydataFolder(data.Dataset):
    """A data loader where the list is arranged in this way:
    
        dog/1.jpg 1
        dog/2.jpg 1
        dog/3.jpg 1
           .
           .
           .
        cat/1.jpg 2
        cat/2.jpg 2
           .
           .
           .
        path      label

    Args:
        
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
        label_shuffling: label shuffle or not. default is False. If want to use label suffling, please modify the dataloader.py, 
                            in the __iter__(), add as follow:
                                        'if callable(self.dataset.__shuffle__):
                                            self.dataset.__shuffle__()'
    """

    def __init__(self, filename, root=None, transform=None, label_shuffling=False):

        if label_shuffling:
            lists, dicts, max_number = read_data_file_with_dict(filename, root)
        else:
            lists = read_data_file(filename, root)
        if len(lists) == 0:
            raise(RuntimeError('Found 0 images in subfolders\n'))

        self.root = root
        self.transform = transform
        self.lists = lists
        self.load = pil_loader
        self.label_shuffling = label_shuffling

        if self.label_shuffling:
            self.dicts = dicts
            self.max_number = max_number
            self.idx = [i for i in range(max_number * len(self.dicts))]
        else:
            self.idx = [i for i in range(len(self.lists))]

    def __shuffle__(self):

        if self.label_shuffling:
            label_shuffle(self.dicts, self.max_number, self.idx)
            print self.idx

    def __getitem__(self, index):
        """
        Args:
            index (int): index

        Returns:
            tuple: (image, label) where label is the clas of the image
        """

        path, label = self.lists[self.idx[index]]
        img = self.load(path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        
        return len(self.idx)

class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, label, topk=(1,)):

    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step'):

    if policy == 'fixed':
        lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')

def Config(filename):

    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print '{}: {}'.format(x, parser[x])
    return parser
