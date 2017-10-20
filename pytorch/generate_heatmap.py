import os
import json
import sys
import argparse
import torch
import time
import numpy as np
import cv2
from scipy.misc import imresize
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from inceptionresnetv2 import inceptionresnetv2 as inceptionresv2

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
    parser.add_argument('--num_classes', type=int,
                        dest='num_classes', help='the number of class')
    parser.add_argument('--topk', default=5, type=int,
                        dest='topk', help='topk accuracy (default: 5)')

    return parser.parse_args()

def hook_feature(module, input, output):

    global features
    features.append(np.squeeze(output.data.cpu().numpy()))

def load_model(args):

    model, settings = inceptionresv2(num_classes=1000)
    model.classif = nn.Linear(1536, args.num_classes)

    state_dict = torch.load(args.model)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
        # name = k[7:]
        # new_state_dict[name] = v
    model.load_state_dict(state_dict)
    model.eval()
    features_name = 'conv2d_7b'
    model.conv2d_7b.register_forward_hook(hook_feature)

    model = model.cuda(args.gpu[0])

    return model, settings

def get_weight(model):

    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    
    return weight_softmax

def get_heatmap(feature, weight, class_idx, top_k):

    size_upsample = (299, 299)
    nc, h, w = feature.shape
    heat_map = np.zeros(size_upsample)
    cnt = 0
    for idx in class_idx:
        cam = weight[idx].dot(feature.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        heat_map += imresize(cam_img, size_upsample)
        cnt += 1
        if cnt == top_k:
            break

    heat_map /= top_k
    heat_map = np.uint8(heat_map)
    return heat_map

def generate_heatmap(model, settings, weight, args):

    global features

    fp = open(args.test_file, 'r')
    lines = fp.readlines()
    fp.close()

    normalize = transforms.Normalize(mean=settings['mean'],
                                    std=settings['std'])

    size = 299
    transformer = transforms.Compose([transforms.Scale((size, size)),
                transforms.ToTensor(),
                normalize,
            ])

    cc = 0

    for line in lines:
        features = []
        
        info = line.strip().split(' ')
        label = int(info[1])
        img = Image.open(os.path.join(args.root, info[0]))
        input = torch.autograd.Variable(transformer(img).unsqueeze(0).cuda(args.gpu[0]), volatile=True)
        logit = model.forward(input)
        h_x = F.softmax(logit).data.squeeze()
        probs, idx = h_x.sort(0, True)
        if idx[0] == label or idx[1] == label or idx[2] == label:
            continue

        heat_map = get_heatmap(features[0], weight, idx, args.topk)
        test_img = cv2.imread(os.path.join(args.root, info[0]))
        height, width,_ = test_img.shape
        heatmap = cv2.applyColorMap(cv2.resize(heat_map, (width, height)), cv2.COLORMAP_JET)

        result = heatmap * 0.4 + test_img * 0.5
        cv2.imwrite('heatmap_{}_{}_{}'.format(label, idx[0], info[0]), result)
        cc += 1
        if cc == 10:
            break

if __name__ == '__main__':

    args = parse()
    model, settings = load_model(args)
    weight = get_weight(model)
    generate_heatmap(model, settings, weight, args)
