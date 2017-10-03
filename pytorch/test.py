import os
import sys
import argparse
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models
import torchvision.transforms as transforms
from BasicTool import accuracy as accuracy
from BasicTool import MydataFolder as MydataFolder
from BasicTool import AverageMeter as AverageMeter

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        dest='model', help='the path of model to be tested')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--test_file', type=str,
                        dest='test_file', help='the path of test file')
    parser.add_argument('--workers', default=6, type=int,
                        dest='workers', help='the number of data loading workers (default: 6)')
    parser.add_argument('--batch_size', default=64, type=int,
                        dest='batch_size', help='batch_size (default: 64)')
    parser.add_argument('--topk', default=5, type=int,
                        dest='topk', help='topk accuracy (default: 5)')

    return parser.parse_args()

def construct_model(args):

    resnet50 = models.resnet50()
    resnet50.fc = nn.Linear(2048, 80)
    state_dict = torch.load(args.model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    resnet50.load_state_dict(new_state_dict)
    resnet50.avgpool = nn.AvgPool2d(10)
    resnet50 = torch.nn.DataParallel(resnet50, device_ids=args.gpu).cuda(args.gpu[0])

    return resnet50

def test(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
            MydataFolder(args.test_file, args.root,
                transforms.Compose([transforms.Scale(320),
                    transforms.CenterCrop(320),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    top1 = AverageMeter()
    topk = AverageMeter()

    model.eval()
    
    for i, (input, label) in enumerate(test_loader):
        label = label.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        label_var = torch.autograd.Variable(label, volatile=True)
    
        output = model(input_var)
    
        prec1, preck = accuracy(output.data, label, topk=(1, args.topk))
        top1.update(prec1[0], input.size(0))
        topk.update(preck[0], input.size(0))

    print(
        'Test Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
        'Prec@{0} {topk.val:.3f}% ({topk.avg:.3f}%)\n'.format(
        args.topk, top1=top1, topk=topk))

if __name__ == '__main__':

    args = parse()
    test(construct_model(args), args)
