# -*- coding=utf-8 -*-
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_scatter(xs, ys, labels, category_num):
    colors = cm.rainbow(np.linspace(0, 1, category_num))
    colors[:, 3] = 0.5  # 调整透明度
    labels = np.array(labels)
    xs = np.array(xs)
    ys = np.array(ys)
    f, ax = plt.subplots(1)
    for i in range(category_num):
        cur_xs = xs[labels == i]
        cur_ys = ys[labels == i]
        plt.scatter(cur_xs, cur_ys, c=colors[i], label=i)
    ax.legend()
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

def plot_scatter3D(xs, ys, zs, labels, category_num):
    colors = cm.rainbow(np.linspace(0, 1, category_num))
    colors[:, 3] = 0.5  # 调整透明度
    labels = np.array(labels)
    xs = np.array(xs)
    ys = np.array(ys)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(category_num):
        cur_xs = xs[labels == i]
        cur_ys = ys[labels == i]
        cur_zs = zs[labels == i]
        ax.scatter(cur_xs, cur_ys, cur_zs, c=colors[i], label=i)
    ax.legend()
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()