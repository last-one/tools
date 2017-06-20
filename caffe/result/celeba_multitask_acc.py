import os
import numpy as np
import sys

label_file = open('/home/hypan/data/celebA/test.txt', 'r')
lines = label_file.readlines()
label_file.close()

acc = np.zeros(40)
cou = 0

for line in lines:
    info = line.strip('\r\n').split()
    name = info[0].split('.')[0]
    gt_labels = info[1: ]
    feat_path = '/home/hypan/data/celebA/result/' + sys.argv[1] + '/test_feature/' + name + '.npy'
    if not os.path.exists(feat_path):
        print '{} has not predict feature.'.format(name)
    pd_labels = np.load(feat_path)
    cnt = len(pd_labels)
    for i in range(cnt):
        gt_label = int(gt_labels[i])
        pd_label = pd_labels[i]
        if pd_label > 0:
            pd_label = 1
        else:
            pd_label = -1
        if gt_label == pd_label:
            acc[i] += 1
    cou += 1

for i in range(40):
    print i, acc[i] * 1.0 / cou
