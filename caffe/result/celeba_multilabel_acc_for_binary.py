import os
import sys
import struct
import numpy as np

label_file = open('/home/hypan/data/celebA/test.txt', 'r')
lines = label_file.readlines()
label_file.close()

binfile = open(sys.argv[1], 'rb')

acc = np.zeros(40)
cou = 0

for line in lines:
    info = line.strip('\r\n').split()
    gt_labels = info[1: ]
    cnt = len(gt_labels)
    for i in range(cnt):
        gt_label = int(gt_labels[i])
        pd_label = struct.unpack("f", binfile.read(4))[0]
        if pd_label >= 0:
            pd_label = 1
        else:
            pd_label = -1
        if gt_label == pd_label:
            acc[i] += 1
    cou += 1

binfile.close()

for i in range(40):
    print i, acc[i] * 1.0 / cou
