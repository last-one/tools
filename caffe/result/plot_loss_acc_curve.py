import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', dest='log', help='the training log')
    parser.add_argument('-o', '--output', dest='output_path', help='the path to save the picture', type=str, default=None)
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    logs = open(args.log, 'r')
    lines = logs.readlines()
    logs.close()

    name = args.log.split('/')[-1].split('.')[0] + '.jpg'
    if args.output_path != None:
        name = os.path.join(args.output_path, name)

    train_loss = []
    test_acc = []
    max_iter = 0
    display = 0
    test_interval = -1

    for line in lines:
        if line.find('Iteration') == -1 or line.find('loss = ') == -1:
            continue
        st_iter = line.find('Iteration')
        ed_iter = st_iter + 10 + line[st_iter + 10:].find(' ')
        display = max_iter
        max_iter = int(line[st_iter + 9: ed_iter])
        display = max_iter - display

        pos_loss = line.find('loss = ')
        loss = float(line[pos_loss + 7: ])
        train_loss.append(loss)
    max_iter += display

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(0, max_iter, display), train_loss)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    if test_interval != -1:
        ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
        ax2.set_ylabel('test accuracy')
    plt.savefig(name)
