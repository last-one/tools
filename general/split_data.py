import os
import shutil
import argparse
import numpy.random as rd

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', dest='label', help='the path of label')
    parser.add_argument('-o', '--output', dest='output_path', help='the path of output')
    parser.add_argument('-r', '--ratio', dest='ratio', help='the ratio of training data in all data, default is 0.8', type=float, default=0.8)
    parser.add_argument('-i', '--input', dest='input_path', help='the path of image')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    fp = open(args.label, 'r')
    lines = fp.readlines()
    fp.close()

    train_path = os.path.join(args.output_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    train_label = open(os.path.join(args.output_path,'train.txt'), 'w')

    val_path = os.path.join(args.output_path, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    val_label = open(os.path.join(args.output_path,'val.txt'), 'w')

    for line in lines:
        info = line.strip('\r\n').split()
        name = info[0]
        num = rd.uniform(0, 1)
        if num < args.ratio:
            train_label.write(line)
            shutil.copy(os.path.join(args.input_path, name), os.path.join(train_path, name))
        else:
            val_label.write(line)
            shutil.copy(os.path.join(args.input_path, name), os.path.join(val_path, name))

    train_label.close()
    val_label.close()
