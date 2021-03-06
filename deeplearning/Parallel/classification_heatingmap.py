# -*- coding=utf-8 -*-
import numpy as np
from PIL import Image
import os
from utils.Tools import calculate_acc_error
class_num = 4


def get_label_from_pixelvalue(pixel_value):
    '''
    根据像素值返回预测的label的值
    :param pixel_value:
    :return:
    '''
    if pixel_value[0] >= 200 and pixel_value[1] >= 200 and pixel_value[2] >= 200:
        print 'Error'
        return 1
    if pixel_value[1] >= 200 and pixel_value[2] >= 200:
        return 3
    if pixel_value[0] >= 200:
        return 2
    if pixel_value[1] >= 200:
        return 0
    if pixel_value[2] >= 200:
        return 1


def generate_feature_by_heatingmap(image):
    '''
    产生一幅热力图对应的特征向量
    :param image:　热力图
    :return:对应的特征向量
    '''
    features = np.zeros([1, class_num], np.float32)
    shape = list(np.shape(image))
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel_value = image[i, j]
            index = get_label_from_pixelvalue(pixel_value)
            features[0, index] += 1
    features /= np.sum(features)
    return np.array(features).squeeze()


def generate_features_multiheatingmap(dir_path):
    names = os.listdir(dir_path)
    image_paths = [os.path.join(dir_path, name) for name in names]
    features = [generate_feature_by_heatingmap(np.array(Image.open(path))) for path in image_paths]
    return features


def generate_features_labels(data_dir):
    '''
    生成data_dir目录下面所有文件的features以及对应的labels
    :param data_dir:
    :return:
    '''
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    test_features = []
    test_labels = []
    for subclass in ['train', 'val', 'test']:
        for type in [0, 1, 2, 3]:
            cur_features = generate_features_multiheatingmap(os.path.join(data_dir, subclass, str(type)))
            if subclass == 'train':
                train_features.extend(cur_features)
                train_labels.extend([type] * len(cur_features))
            elif subclass == 'val':
                val_features.extend(cur_features)
                val_labels.extend([type] * len(cur_features))
            else:
                test_features.extend(cur_features)
                test_labels.extend([type] * len(cur_features))
    scio.savemat('data.mat', {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels
    })
    return train_features, train_labels, val_features, val_labels, test_features, test_labels

if __name__ == '__main__':
    # 从生成的图片中提取特征
    # train_features, train_labels, val_features, val_labels, test_features, test_labels = \
    #     generate_features_labels('/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/heatingmap/ICIP/4-classes/patch_ROI')

    import scipy.io as scio
    from utils.classification import SVM, KNN
    crossid = 1
    train_data = scio.loadmat('./features/crossvalidation/' + str(crossid) + '/train.npy.mat')
    train_features = train_data['features']
    train_labels = train_data['labels']

    val_data = scio.loadmat('./features/crossvalidation/' + str(crossid) + '/val.npy.mat')
    val_features = val_data['features']
    val_labels = val_data['labels']

    test_data = scio.loadmat('./features/crossvalidation/' + str(crossid) + '/test.npy.mat')
    test_features = test_data['features']
    test_labels = test_data['labels']
    # SVM
    predicted_label, c_params, g_params, accs = SVM.do(train_features, train_labels, val_features, val_labels,
                                                       adjust_parameters=True)

    predicted_label, acc = SVM.do(train_features, train_labels, test_features, test_labels, adjust_parameters=False,
                                  C=c_params, gamma=g_params)
    print 'ACC is ', acc
    calculate_acc_error(predicted_label, test_labels)