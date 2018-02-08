# -*- coding=utf-8
from glob import glob
import scipy.io as scio
import os
import numpy as np
from PIL import Image
from utils.Tools import shuffle_image_label, read_mhd_image, get_boundingbox, convert2depthlaster
from Config import Config as net_config
import tensorflow as tf
from model import inference
import shutil

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('load_model_path', '/home/give/PycharmProjects/MICCAI2018/deeplearning/Parallel/parameters/0',
                           '''the model reload path''')
tf.app.flags.DEFINE_string('save_model_path', './models', 'the saving path of the model')
tf.app.flags.DEFINE_string('log_dir', './log/train',
                           """The Summury output directory""")
tf.app.flags.DEFINE_string('log_val_dir', './log/val',
                           """The Summury output directory""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('max_steps', 1000000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
MOMENTUM = 0.9
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.squeeze(logits),
                                                                   labels=tf.squeeze(labels))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('loss', loss_)

    return loss_


def load_ROI(dir_name):
    phasenames = ['NC', 'ART', 'PV']
    mask_images = []
    mhd_images = []
    for phasename in phasenames:
        image_path = glob(os.path.join(dir_name, phasename + '_Image*.mhd'))[0]
        mask_path = os.path.join(dir_name, phasename + '_Registration.mhd')
        mhd_image = read_mhd_image(image_path, rejust=True)
        mhd_image = np.squeeze(mhd_image)
        # show_image(mhd_image)
        mask_image = read_mhd_image(mask_path)
        mask_image = np.squeeze(mask_image)
        [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
        # xmin -= 15
        # xmax += 15
        # ymin -= 15
        # ymax += 15
        mask_image = mask_image[xmin: xmax, ymin: ymax]
        mhd_image = mhd_image[xmin: xmax, ymin: ymax]
        mhd_image[mask_image != 1] = 0
        mask_images.append(mask_image)
        mhd_images.append(mhd_image)
    mhd_images = convert2depthlaster(mhd_images)
    return mhd_images





class DataSet:
    @staticmethod
    def load_liver_density(data_dir='/home/give/PycharmProjects/MedicalImage/Net/forpatch/ResNetMultiPhaseExpand'):
        '''
        加载调整过窗宽窗位的肝脏平均密度
        :param data_dir: mat文件的路径
        :return:dict类型的对象，key是我们的文件名，value是长度为３的数组代表的是三个Ｐｈａｓｅ的平均密度
        '''
        mat_paths = glob(os.path.join(data_dir, 'liver_density*.mat'))
        total_liver_density = {}
        for mat_path in mat_paths:
            liver_density = scio.loadmat(mat_path)
            for (key, value) in liver_density.items():
                if key.startswith('__'):
                    continue
                if key in total_liver_density.keys():
                    print 'Error', key
                total_liver_density[key] = np.array(value).squeeze()
        return total_liver_density

    @staticmethod
    def load_raw_liver_density(data_dir='/home/give/PycharmProjects/MedicalImage/Net/forpatch/ResNetMultiPhaseExpand'):
        '''
        加载原生的肝脏平均密度
        :param data_dir: mat文件的路径
        :return:dict类型的对象，key是我们的文件名，value是长度为３的数组代表的是三个Ｐｈａｓｅ的平均密度
        '''
        mat_paths = glob(os.path.join(data_dir, 'raw_liver_density*.mat'))
        total_liver_density = {}
        for mat_path in mat_paths:
            liver_density = scio.loadmat(mat_path)
            for (key, value) in liver_density.items():
                if key.startswith('__'):
                    continue
                if key in total_liver_density.keys():
                    print 'Error', key
                total_liver_density[key] = np.array(value).squeeze()
        return total_liver_density

    @staticmethod
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

    @staticmethod
    def generate_paths(dir_name, target_labels=[0, 1, 2, 3], mapping_label={0: 0, 1: 1, 2: 2, 3: 3}, shuffle=True):
        '''
        返回dirname中的所有病灶图像的路径
        :param dir_name:  父文件夹的路径
        :param cross_ids: 包含的交叉的折，一般来说我们做三折交叉验证,cross_ids就是[0, 1] 或者是[2]
        :param target_labels: 需要文件标注的label
        :return:
        '''
        roi_paths = []
        labels = []
        cur_dir = dir_name
        print cur_dir
        names = os.listdir(cur_dir)
        for name in names:
            if int(name[-1]) not in target_labels:
                continue
            type_dir = os.path.join(cur_dir, name)
            roi_paths.append(type_dir)
            labels.append(mapping_label[int(name[-1])])
        if shuffle:
            roi_paths, labels = shuffle_image_label(roi_paths, labels)
        return roi_paths, roi_paths, labels

    def __init__(self, data_dir, state, pre_load=False, divied_liver=False, rescale=True):
        '''
        DataSet的初始化函数
        :param data_dir: 数据的文件夹，存储ｐａｔｃｈ的路径　一般结构是data_dir/train/0/*
        :param state:
        :param pre_load: 是否将数据全部提前加载进来, 默认是Ｆａｌｓｅ
        :param divied_liver: 是否除以肝脏的平均密度
        :param rescale: 是否将像素值进行放缩，放缩到[-1, 1]之间
        :param full_roi_path: 完整的存储ｒｏｉ的路径，不是存储ｐａｔｃｈ的路径
        :param expand_is_roi: 决定ｅｘｐａｎｄ是否是ＲＯＩ　如果是则ｅｘｐａｎｄ代表的是完整的ＲＯＩ，否则ｅｘｐａｎｄ就是同一个ｐａｔｃｈ放缩的不同的ｓｃａｌｅ
        '''
        self.roi_paths, self.expand_roi_path, self.labels = DataSet.generate_paths(
            data_dir
        )
        if self.roi_paths[0].endswith('.npy'):
            self.using_raw = True
        else:
            self.using_raw = False
        self.state = state
        self.epoch_num = 0
        self.start_index = 0
        # self.liver_density = DataSet.load_liver_density()   # 调整过窗宽窗位的肝脏平均密度
        # self.raw_liver_density = DataSet.load_raw_liver_density()   # 原始的像素值的平均密度
        # print self.raw_liver_density
        self.divied_liver = divied_liver
        self.rescale = rescale

        self.pre_load = pre_load
        if self.pre_load:
            self.ROIs_images = [np.asarray(load_ROI(path)) for path in self.roi_paths]

    def get_next_batch(self, batch_size):
        if not self.pre_load:
            while True:
                cur_expand_roi_paths = []
                cur_labels = []
                end_index = self.start_index + batch_size
                if end_index > len(self.roi_paths):
                    self.epoch_num += 1

                    cur_expand_roi_paths.extend(self.expand_roi_path[self.start_index: len(self.roi_paths)])
                    cur_expand_roi_paths.extend(self.expand_roi_path[:end_index - len(self.roi_paths)])

                    cur_labels.extend(self.labels[self.start_index: len(self.roi_paths)])
                    cur_labels.extend(self.labels[:end_index - len(self.roi_paths)])
                    self.start_index = end_index - len(self.roi_paths)
                    print 'state: ', self.state, ' epoch: ', self.epoch_num

                else:
                    cur_expand_roi_paths.extend(self.expand_roi_path[self.start_index: end_index])
                    cur_labels.extend(self.labels[self.start_index: end_index])
                    self.start_index = end_index
                cur_expand_roi_images = [
                    np.asarray(load_ROI(path)) for path in
                    cur_expand_roi_paths]
                cur_expand_roi_images = DataSet.resize_images(cur_expand_roi_images, net_config.ROI_SIZE_W, self.rescale)
                # print np.shape(cur_roi_images)
                yield cur_expand_roi_images, cur_labels
        else:
            while True:
                cur_rois_images = []
                cur_labels = []
                end_index = self.start_index + batch_size
                if end_index > len(self.roi_paths):
                    self.epoch_num += 1

                    cur_rois_images.extend(self.ROIs_images[self.start_index: len(self.roi_paths)])
                    cur_rois_images.extend(self.ROIs_images[:end_index - len(self.roi_paths)])

                    cur_labels.extend(self.labels[self.start_index: len(self.roi_paths)])
                    cur_labels.extend(self.labels[:end_index - len(self.roi_paths)])
                    self.start_index = end_index - len(self.roi_paths)
                    print 'state: ', self.state, ' epoch: ', self.epoch_num

                else:
                    cur_rois_images.extend(self.ROIs_images[self.start_index: end_index])
                    cur_labels.extend(self.labels[self.start_index: end_index])
                    self.start_index = end_index
                cur_rois_images = DataSet.resize_images(cur_rois_images, net_config.EXPAND_SIZE_W, self.rescale)
                yield cur_rois_images, cur_labels


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size


def calculate_accuracy(logits, labels, arg_index=1):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, arg_index), tf.arg_max(labels, arg_index))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope+'/accuracy', accuracy)
    return accuracy


def train(logits, images_tensor, labels_tensor, is_training_tensor, iterator_num, summary_path='./log', restore=None):
    cross_id = 1
    roi_dir = '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation'
    pre_load = True
    train_dataset = DataSet(os.path.join(roi_dir, str(cross_id), 'train'), 'train', pre_load=pre_load,
                            rescale=True, divied_liver=False)
    val_dataset = DataSet(os.path.join(roi_dir, str(cross_id), 'val'), 'val', pre_load=pre_load,
                          rescale=True, divied_liver=False)
    train_batchdata = train_dataset.get_next_batch(net_config.BATCH_SIZE)
    val_batchdata = val_dataset.get_next_batch(net_config.BATCH_SIZE)

    predicted_tensor = tf.argmax(logits, 1)
    global_step_tensor = tf.Variable(initial_value=0, trainable=False)
    softmax_loss = loss(logits, labels_tensor)
    loss_tensor = softmax_loss
    tf.summary.scalar('softmax loss', softmax_loss)
    tf.summary.scalar('loss', loss_tensor)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_tensor, global_step=global_step_tensor)
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op('train')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(tf.squeeze(labels_tensor), tf.int64))
    accuracy_tensor = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy_tensor)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if restore is not None:
            full_path = tf.train.latest_checkpoint(restore['path'])
            print 'load model from ', full_path
            saver.restore(sess, full_path)
        train_summary_writer = tf.summary.FileWriter(os.path.join(summary_path, 'train'), graph=sess.graph)
        val_summary_writer = tf.summary.FileWriter(os.path.join(summary_path, 'val'), graph=sess.graph)
        merged_summary_op = tf.summary.merge_all()
        for i in range(iterator_num):
            step_value = sess.run(global_step_tensor)

            train_expand_roi_batch_images, train_labels = train_batchdata.next()
            # 反向传播的同时更新center value
            _, train_acc, train_prediction, loss_value, merged_summary_value, softmax_loss_value = sess.run(
                [train_op, accuracy_tensor, predicted_tensor, loss_tensor, merged_summary_op,
                 softmax_loss], feed_dict={
                    images_tensor: train_expand_roi_batch_images,
                    labels_tensor: train_labels,
                    is_training_tensor: True,
                })
            train_summary_writer.add_summary(merged_summary_value, global_step=step_value)
            if step_value % 1000 == 0:
                val_expand_roi_batch_images, val_labels = val_batchdata.next()
                validation_acc, loss_value, merged_summary_value = sess.run(
                    [accuracy_tensor, loss_tensor, merged_summary_op],
                    feed_dict={
                        images_tensor: val_expand_roi_batch_images,
                        labels_tensor: val_labels,
                        is_training_tensor: False,
                    })
                val_summary_writer.add_summary(merged_summary_value, step_value)
                print 'step: %d, validation accuracy: %.2f, validation loss: %.2f' % (
                    step_value, validation_acc, loss_value)
                save_model_path = os.path.join('./parameters/', str(cross_id))
                checkpoint_path = os.path.join(save_model_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step_tensor)
                save_dir = os.path.join(save_model_path, str(step_value))
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                filenames = glob(os.path.join(save_model_path, '*-' + str(int(step_value + 1)) + '.*'))
                for filename in filenames:
                    shutil.copy(
                        filename,
                        os.path.join(save_dir, os.path.basename(filename))
                    )
            if step_value % 100 == 0:
                print 'step: %d, training accuracy: %.2f, training loss: %.2f, softmax_loss_value: %.2f' % (
                    step_value, train_acc, loss_value, softmax_loss_value)
                # print centers_value
        train_summary_writer.close()
        val_summary_writer.close()


if __name__ == '__main__':
    roi_dir = '/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation'
    pre_load = True
    cross_id = 0
    category_num = 4
    is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
    train_dataset = DataSet(os.path.join(roi_dir, str(cross_id), 'train'), 'train', pre_load=pre_load,
                            rescale=True, divied_liver=False)
    train_batchdata = train_dataset.get_next_batch(net_config.BATCH_SIZE)
    roi_images, labels = train_batchdata.next()
    print np.shape(roi_images)