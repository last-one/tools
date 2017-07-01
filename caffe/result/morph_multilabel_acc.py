import os
import numpy as np
import sys
import math

label_file = open('/home/hypan/data/morph/val.txt', 'r')
lines = label_file.readlines()
label_file.close()

acc = np.zeros(5)
cou = 0
label = ['gender', 'age', 'race']

for line in lines:
    info = line.strip('\r\n').split()
    name = info[0].split('.')[0]
    gt_labels = info[1: ]
    cnt = len(gt_labels)
    for i in range(cnt):
        gt_label = int(gt_labels[i])
        pd_label = np.load(os.path.join(sys.argv[1], name + "_" + label[i] + '.npy'))
        ids = np.argmax(pd_label)
        if i != 1:
            if gt_label == ids:
                acc[i] += 1
            # else:
                # sums = 0
                # for p in pd_label:
                    # sums += math.exp(p)
                # print math.exp(pd_label[ids]) / sums, math.exp(pd_label[gt_label]) / sums
        else:
            if abs(int(round(pd_label[ids])) - gt_label) <= 5:
                acc[i] += 1
                acc[3] += abs(int(round(pd_label[ids])) - gt_label)
            acc[4] += abs(int(round(pd_label[ids])) - gt_label)
    cou += 1

for i in range(5):
    print i, acc[i] * 1.0 / cou
