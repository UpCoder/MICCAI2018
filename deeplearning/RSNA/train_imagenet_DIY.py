# -*- coding=utf-8 -*-
from train import train
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
from model import inference
from Config import Config as net_config

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', net_config.BATCH_SIZE, "batch size")


def main(_):
    roi_images = tf.placeholder(
        shape=[
            net_config.BATCH_SIZE,
            net_config.ROI_SIZE_W,
            net_config.ROI_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='roi_input'
    )
    labels_tensor = tf.placeholder(
        shape=[
            net_config.BATCH_SIZE
        ],
        dtype=np.int32
    )
    is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[])
    logits= inference(
        roi_images,
        is_training=is_training_tensor
        )
    restore_obj = dict()
    restore_obj['path'] = '/home/give/PycharmProjects/MICCAI2018/deeplearning/RSNA/parameters/1'
    train(logits, roi_images, labels_tensor, is_training_tensor, int(1e6), restore=None)

if __name__ == '__main__':
    tf.app.run()
