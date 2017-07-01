import os
import sys
import struct
import numpy as np

label_file = open('/home/hypan/data/webface/test.txt', 'r')
lines = label_file.readlines()
label_file.close()

binfile = open(sys.argv[1], 'rb')

acc = 0
cou = 0

for line in lines:
    info = line.strip('\r\n').split()
    gt_label = int(info[1])
    pd_labels = struct.unpack("10575f", binfile.read(4*10575))
    pd_labels = np.array(pd_labels)
    ids = pd_labels.argmax()
    if gt_label == ids:
        acc += 1
    cou += 1

binfile.close()

print acc * 1.0 / cou
