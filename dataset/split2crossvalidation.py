# -*- coding=utf-8 -*-
import os
import numpy as np
import shutil
from utils.Tools import findStr


def read_all_names(dir_name):
    '''
    读取所有文件的路径名
    :param dir_name: 总的目录，一般来说下面有三个子目录：train，test，val
    :return: 返回的字典，key是pclid，value是数组，每个元素对应的每个Slice的文件夹
    '''
    all_paths = []
    for sub_class in ['train', 'test', 'val']:
        cur_dir = os.path.join(dir_name, sub_class)
        names = os.listdir(cur_dir)
        paths = [os.path.join(cur_dir, name) for name in names]
        all_paths.extend(paths)
    res_dict = {}
    for path in all_paths:
        basename = os.path.basename(path)
        pclid = basename[: findStr(basename, '_', 3)]  # 病人_检查号_病灶ID
        if pclid in res_dict.keys():
            res_dict[pclid].append(path)
        else:
            res_dict[pclid] = []
            res_dict[pclid].append(path)
    return res_dict


def split_random(dict_paths, train_dir, val_dir, test_dir):
    '''
    将上述得到的所有文件随机划分成训练集、验证集、测试集
    :param dict_paths:read_all_names得到的字典类型的数据
    :param train_dir:训练集所在的目录
    :param val_dir:验证集所在的目录
    :param test_dir:测试集所在的目录
    :return:
    '''
    for pclid in dict_paths.keys():
        random = np.random.random()
        if random < 0.8:
            # train or val
            if random < 0.2:
                # move to val
                for path in dict_paths[pclid]:
                    target_dir = os.path.join(val_dir, os.path.basename(path))
                    # if not os.path.exists(target_dir):
                    #     os.makedirs(target_dir)
                    shutil.copytree(path, target_dir)
            else:
                # move to train
                for path in dict_paths[pclid]:
                    target_dir = os.path.join(train_dir, os.path.basename(path))
                    # if not os.path.exists(target_dir):
                    #     os.makedirs(target_dir)

                    shutil.copytree(path, target_dir)
        else:
            # move to test
            for path in dict_paths[pclid]:
                target_dir = os.path.join(test_dir, os.path.basename(path))
                # if not os.path.exists(target_dir):
                #     os.makedirs(target_dir)
                shutil.copytree(path, target_dir)


def statics_in_dir(dir_path):
    '''
    统计一个目录下面病灶类型的分布
    :param dir_path: 目录的路径
    :return: None
    '''
    names = os.listdir(dir_path)
    distribution = [0, 0, 0, 0, 0]
    for name in names:
        distribution[int(name[-1])] += 1
    print os.path.basename(dir_path), ': ', distribution


def statics_in_dirs(dir_path):
    '''
    统计该目录下面train, val, test三个子目录不同类型病灶的分布
    :param dir_path:
    :return:
    '''
    for sub_name in ['train', 'val', 'test']:
        statics_in_dir(
            os.path.join(dir_path, sub_name)
        )

if __name__ == '__main__':
    names = read_all_names('/home/give/Documents/dataset/MICCAI2018/Slices/un-crossvalidation')
    print len(names)
    split_random(names, '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/1/train',
                 '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/1/val',
                 '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/1/test')

    statics_in_dirs('/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/1')
