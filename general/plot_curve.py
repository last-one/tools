import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import spline

def _get_express(strs, i, lists):

    cnt = 0

    for name in lists:
        if cnt != 0:
            strs = strs + ','
        if name == 'marker':
            strs = strs + ' cond[\'{}\'][{}]'.format(name, i)
        else:
            strs = strs + name + ' = cond[\'{}\'][{}]'.format(name, i)
        cnt += 1
    strs = strs + ')'

    return strs

def plot_curve(x, y, xlim=None, ylim=None, xmargin=0, ymargin=0,
               save_path=None, xlabel=None, ylabel=None, title=None,
               markers=None, colors=None, labels=None, grid=False, legend=0):

    x = np.array(x)
    y = np.array(y)

    x_shape = x.shape
    y_shape = y.shape

    assert len(x_shape) == 1 or (x_shape == y_shape)

    if markers != None:
        if len(y_shape) != 1:
            assert y_shape[0] <= len(markers)
        else:
            assert len(markers) == 1

    if colors != None:
        if len(y_shape) != 1:
            assert y_shape[0] <= len(colors)
        else:
            assert len(colors) == 1

    if labels != None:
        if len(y_shape) != 1:
            assert y_shape[0] <= len(labels)
        else:
            assert len(labels) == 1
        if legend == -1:
            legend = 4

    cond = {}
    name_list = []

    if markers != None:
        name_list.append('marker')
        cond['marker'] = markers
    if colors != None:
        name_list.append('color')
        cond['color'] = colors
    if labels != None:
        name_list.append('label')
        cond['label'] = labels

    ax = plt.subplot(111)
    if xmargin != 0:
        ax.xaxis.set_major_locator(MultipleLocator(xmargin))
    if ymargin != 0:
        ax.yaxis.set_major_locator(MultipleLocator(ymargin))

    if xlim != None:
        plt.xlim(xlim[0], xlim[1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])

    if len(y_shape) == 1:
        y = [y]

    cnt = 0
    if len(x_shape) == 1:
        for info in y:
            info = [x, info]
            eval(_get_express('plt.plot(info[0], info[1], ', cnt, name_list))
            cnt += 1
    else:
        for info in zip(x, y):
            eval(_get_express('plt.plot(info[0], info[1], ', cnt, name_list))
            cnt += 1

    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    if title != None:
        plt.title(title)
    if labels != None:
        plt.legend(loc=legend)
    plt.grid(grid)
    if save_path != None:
        plt.savefig(save_path)

def plot_smooth_curve(x, y, xlim=None, ylim=None, xmargin=0, ymargin=0,
               save_path=None, xlabel=None, ylabel=None, title=None,
               markers=None, colors=None, labels=None, grid=False, legend=0):

    x = np.array(x)
    y = np.array(y)

    x_shape = x.shape
    
    if len(x_shape) == 1:
        dot_num = (x.max() - x.min()) * 50
        new_x = np.linspace(x.min(), x.max(), dot_num)
        new_y = []
        for info in y:
            smooth_y = spline(x, info, new_x)
            new_y.append(smooth_y)
    else:
        new_x = []
        new_y = []
        for info in zip(x, y):
            dot_num = (info[0].max() - info[1].max()) * 50
            smooth_x = np.linspace(info[0].min(), info[1].max(), dot_num)
            smooth_y = spline(info[0], smooth_x, info[1])
            new_x.append(smooth_x)
            new_y.append(smooth_y)

    plot_curve(new_x, new_y, xlim, ylim, xmargin, ymargin, save_path, xlabel, ylabel, title, markers, colors, labels, grid, legend)


if __name__ == '__main__':

    fp = open(sys.argv[1], 'r')
    lines = fp.readlines()
    fp.close()

    colors = ['r', 'g', 'b', 'y', 'c', 'k', 'm', 'w']
    marker = ['-o', '-^', '-*', '-D', '-s', '-p', '-x', '-+']
    method = ['ours', 'softmax and contrastive loss', 'only softmax loss']

    num = 11
    xx = np.arange(num)
    cnt = 0
    accs = []
    i = 0
    y = []

    ymajorLocator = MultipleLocator(10)

    ax = plt.subplot(111)
    for line in lines:
        acc = float(line)
        accs.append(acc)
        cnt += 1
        if cnt == num:
            accs = np.array(accs)
            y.append(accs)
            cnt = 0
            accs = []
            i += 1
    
    plot_curve.plot_smooth_curve(xx, y, colors=colors, markers=marker, labels=method, grid=True, save_path='1.pdf', ymargin=10, xlabel='Age error tolerance', ylabel='Cumulative Score (%)')
    plot_curve.plot_smooth_curve(xx, y, colors=colors, labels=method, grid=True, save_path='1.pdf', ymargin=10, xlabel='Age error tolerance', ylabel='Cumulative Score (%)')
