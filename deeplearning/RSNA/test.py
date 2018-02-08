# -*- coding=utf-8 -*-
import tensorflow as tf
from model import inference
import numpy as np
from train import load_ROI
import os
from Config import Config as net_config
from PIL import Image
from utils.Tools import calculate_acc_error


def resize_images(images, size, rescale=True):
    res = np.zeros(
        [
            len(images),
            size,
            size,
            3
        ],
        np.float32
    )
    for i in range(len(images)):
        img = Image.fromarray(np.asarray(images[i], np.uint8))
        # data augment
        random_int = np.random.randint(0, 4)
        img = img.rotate(random_int * 90)
        random_int = np.random.randint(0, 2)
        if random_int == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        random_int = np.random.randint(0, 2)
        if random_int == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        img = img.resize([size, size])
        if rescale:
            res[i, :, :, :] = np.asarray(img, np.float32) / 255.0
            res[i, :, :, :] = res[i, :, :, :] - 0.5
            res[i, :, :, :] = res[i, :, :, :] * 2.0
        else:
            res[i, :, :, :] = np.asarray(img, np.float32)
    return res

def test(images_tensor, labels_tensor, parameters_path, test_data_dir):
    '''
    测试模型的有效性
    :param images_tensor: 图像的tensor
    :param labels_tensor: label的tensor
    :param generator: 数据的生成器，可以通过for (images_batch, labels_batch) in generator:格式来获取数据
    :param parameters_path:模型保存的路径
    :return:
    '''
    logits= inference(images_tensor, False)
    prediction_tensor = tf.argmax(logits, 1)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(tf.squeeze(labels_tensor), tf.int64))
    accuracy_tensor = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    featuremaps = []
    labels = []

    with tf.Session() as sess:
        full_path = tf.train.latest_checkpoint(parameters_path)
        print full_path
        saver.restore(sess, full_path)

        names = os.listdir(test_data_dir)
        paths = []
        labels = []
        all_predictions = []
        for name in names:
            if int(name[-1]) not in [0, 1, 2, 3]:
                continue
            paths.append(os.path.join(test_data_dir, name))
            labels.append(int(name[-1]))
        start_index = 0
        while True:
            end_index = start_index + net_config.BATCH_SIZE
            if end_index > len(paths):
                end_index = len(paths)
            cur_batch_paths = paths[start_index: end_index]
            cur_batch_images = [load_ROI(cur_path) for cur_path in cur_batch_paths]
            cur_batch_labels = labels[start_index: end_index]
            cur_batch_images = resize_images(cur_batch_images, net_config.ROI_SIZE_W, True)
            print np.shape(cur_batch_images)
            print np.shape(cur_batch_labels)
            predicted, accuracy = sess.run([prediction_tensor, accuracy_tensor], feed_dict={
                images_tensor: np.array(cur_batch_images),
                labels_tensor: np.squeeze(cur_batch_labels)
            })
            print accuracy
            all_predictions.extend(predicted)
            start_index = end_index
            if start_index >= len(paths):
                break
    calculate_acc_error(all_predictions, labels)

if __name__ == '__main__':
    roi_images = tf.placeholder(
        shape=[
            None,
            net_config.ROI_SIZE_W,
            net_config.ROI_SIZE_H,
            net_config.IMAGE_CHANNEL
        ],
        dtype=np.float32,
        name='roi_input'
    )
    labels_tensor = tf.placeholder(
        shape=[
            None
        ],
        dtype=np.int32
    )
    restore_obj = dict()
    crossid = 1
    restore_obj['path'] = '/home/give/PycharmProjects/MICCAI2018/deeplearning/RSNA/parameters/' + str(crossid)
    test_data_dir = '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/' + str(crossid) + '/test'
    test(roi_images, labels_tensor, parameters_path=restore_obj['path'], test_data_dir=test_data_dir)