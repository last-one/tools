import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
import argparse
import time
from BasicTool import MydataFolder as MydataFolder 
from BasicTool import adjust_learning_rate as adjust_learning_rate
from BasicTool import AverageMeter as AverageMeter
from BasicTool import accuracy as accuracy
from BasicTool import save_checkpoint as save_checkpoint
from BasicTool import Config as Config

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_file', type=str,
                        dest='train_file', help='the path of train file')
    parser.add_argument('--val_file', default=None, type=str,
                        dest='val_file', help='the path of val file')
    parser.add_argument('--num_classes', default=1000, type=int,
                        dest='num_classes', help='num_classes (default: 1000)')

    return parser.parse_args()

def construct_model(args):

    resnet152 = models.resnet152(True)
    resnet152.fc = nn.Linear(2048, args.num_classes)
    resnet152 = torch.nn.DataParallel(resnet152, device_ids=args.gpu).cuda()

    return resnet152

def train_val(model, args):

    traindir = args.train_file
    valdir = args.val_file

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    config = Config(args.config)
    cudnn.benchmark = True
    
    train_loader = torch.utils.data.DataLoader(
            MydataFolder(traindir, args.root, 
                transforms.Compose([transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

    if config.test_interval != 0 and args.val_file is not None:
        val_loader = torch.utils.data.DataLoader(
                MydataFolder(valdir, args.root,
                    transforms.Compose([transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), config.base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    
    end = time.time()
    iters = 0
    best_model = 0
    learning_rate = config.base_lr

    model.train()
    while iters < config.max_iter:
    
        for i, (input, label) in enumerate(train_loader):

            data_time.update(time.time() - end) 

            label = label.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            label_var = torch.autograd.Variable(label)

            output = model(input_var)
            loss = criterion(output, label_var)
            prec1, preck = accuracy(output.data, label, topk=(1, config.topk))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            topk.update(preck[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data load {data_time.val:.3f} ({data_time.avg:3f})\n'
                    'Learning rate = {1}\n'
                    'Loss = {loss.val:.4f} (ave = {loss.avg:.4f})\t'
                    'Prec@1 = {top1.val:.3f}% (ave = {top1.avg:.3f}%)\t'
                    'Prec@{2} = {topk.val:.3f}% (ave = {topk.avg:.3f}%)\n'
                    '-----------------------------------------------------------------------------------------------------------------'.format(
                    iters, learning_rate, config.topk, batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    top1=top1, topk=topk))
                batch_time.reset()
                data_time.reset()
                losses.reset()
                top1.reset()
                topk.reset()
    
            if config.test_interval != 0 and args.val_file is not None and iters % config.test_interval == 0:

                model.eval()
                for i, (input, label) in enumerate(val_loader):
                    label = label.cuda(async=True)
                    input_var = torch.autograd.Variable(input, volatile=True)
                    label_var = torch.autograd.Variable(label, volatile=True)
    
                    output = model(input_var)
                    loss = criterion(output, label_var)
    
                    prec1, preck = accuracy(output.data, label, topk=(1, config.topk))
                    losses.update(loss.data[0], input.size(0))
                    top1.update(prec1[0], input.size(0))
                    topk.update(preck[0], input.size(0))
    
                batch_time.update(time.time() - end)
                end = time.time()
                is_best = top1.avg > best_model
                best_model = max(best_model, top1.avg)
                save_checkpoint({
                    'iter': iters,
                    'state_dict': model.state_dict(),
                    }, is_best, 'resnet152_places365'.format(iters))
    
                print(
                    'Test Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    'Prec@{0} {topk.val:.3f}% ({topk.avg:.3f}%)\n'
                    '-----------------------------------------------------------------------------------------------------------------'.format(
                    config.topk, batch_time=batch_time,
                    loss=losses, top1=top1, topk=topk))
    
                batch_time.reset()
                losses.reset()
                top1.reset()
                topk.reset()
                
                model.train()
    
            if iters == config.max_iter:
                break
            learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)

if __name__ == '__main__':

    args = parse()
    train_val(construct_model(args), args)
