import os
import numpy as np
from ExtractPatches import extract_patches_multidir
from utils.Tools import calculate_acc_error
from glob import glob

def cal_distance(patches, center):
    '''

    :param patches: None 49
    :param center: 128 * 49
    :return:
    '''
    patches2 = np.multiply(patches, patches)
    center2 = np.multiply(center, center)
    patchdotcenter = np.array(np.dot(np.mat(patches), np.mat(center).T))  # None * 128
    patches2sum = np.sum(patches2, axis=1)  # None
    center2sum = np.sum(center2, axis=1)    # 128
    distance_arr = np.zeros([len(patches2sum), len(center2sum)])
    for i in range(len(patches2sum)):
        for j in range(len(center2sum)):
            distance_arr[i, j] = patches2sum[i] + center2sum[j] - 2 * patchdotcenter[i, j]
    return distance_arr

def load_vocabulary(data_dir):
    return np.load(data_dir)


def generate_representor(data_dir, dictionary_path, subclass, phase_name):
    dictionary = load_vocabulary(dictionary_path)
    shape_vocabulary = np.shape(dictionary)
    vocabulary_size = shape_vocabulary[0]
    representers = []
    patches, coding_labeles, labeles = extract_patches_multidir(data_dir, subclasses=[subclass], return_flag=True,
                                                                phase_name=phase_name)
    all_patches = []
    counts = []
    for case_index, cur_patches in enumerate(patches):
        print np.shape(cur_patches)
        all_patches.extend(cur_patches)
        counts.append(len(cur_patches))
    all_distance_arr = cal_distance(all_patches, dictionary)
    start = 0
    for case_index, count in enumerate(counts):
        distance_arr = all_distance_arr[start: start + count]
        cur_case_representor = np.zeros([1, vocabulary_size])
        for i in range(len(distance_arr)):
            min_index = np.argmin(distance_arr[i])
            cur_case_representor[0, min_index] += 1
        representers.append(cur_case_representor.squeeze())
        start += count
    return representers, labeles


def execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    from LeaningBased.classification import SVM, LinearSVM, KNN
    predicted_label, c_params, g_params, max_c, max_g, accs = SVM.do(train_features, train_labels, val_features,
                                                                     val_labels,
                                                                     adjust_parameters=True)
    predicted_label, accs = SVM.do(train_features, train_labels, test_features, test_labels,
                                   adjust_parameters=False, C=max_c, gamma=max_g)
    calculate_acc_error(predicted_label, test_labels)
    print 'ACA is ', accs
    return accs


def generate_representor_multidir(data_dir, patch_dir, reload=None):
    import scipy.io as scio
    if reload is None:
        train_features, train_labels = generate_representor(data_dir, dictionary_path=patch_dir,
                                                             subclass='train')
        scio.savemat('./training.mat', {
            'features': train_features,
            'labels': train_labels
        })
        test_features, test_labels = generate_representor(data_dir, dictionary_path=patch_dir,
                                                          subclass='test')
        scio.savemat('./testing.mat', {
            'features': test_features,
            'labels': test_labels
        })
        val_features, val_labels = generate_representor(data_dir, dictionary_path=patch_dir, subclass='val')
        scio.savemat('./validation.mat', {
            'features': val_features,
            'labels': val_labels
        })
    else:
        train_data = scio.loadmat('./training.mat')
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = scio.loadmat('./testing.mat')
        test_features = test_data['features']
        test_labels = test_data['labels']

        val_data = scio.loadmat('./validation.mat')
        val_features = val_data['features']
        val_labels = val_data['labels']
    acc = execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels)
    return acc


def generate_representor_multidir_multiphase(data_dir, dictionary_dir):
    def concatenate(all, will_add):
        if all is None:
            all = will_add
        else:
            all = np.concatenate((all, will_add), axis=1)
        return all
    train_features = None
    train_labels = None
    val_features = None
    val_labels = None
    test_features = None
    test_labels = None
    # phase_names = ['ART', 'PV']
    phase_names = ['PV']
    for phase_name in phase_names:
        dictionary_path = glob(os.path.join(dictionary_dir, phase_name+"*.npy"))[0]
        train_features_singlephsae, train_labels_singlephase = generate_representor(data_dir, dictionary_path,
                                                                                    subclass='train', phase_name=phase_name)
        val_features_singlephsae, val_labels_singlephase = generate_representor(data_dir, dictionary_path,
                                                                                subclass='val', phase_name=phase_name)
        test_features_singlephsae, test_labels_singlephase = generate_representor(data_dir, dictionary_path,
                                                                                  subclass='test', phase_name=phase_name)
        train_features = concatenate(train_features, train_features_singlephsae)
        val_features = concatenate(val_features, val_features_singlephsae)
        test_features = concatenate(test_features, test_features_singlephsae)
        if train_labels is None:
            train_labels = train_labels_singlephase
        else:
            if np.sum(np.not_equal(train_labels, train_labels_singlephase)) != 0:
                print train_labels
                print train_labels_singlephase
                print 'Error, not equal'
        if val_labels is None:
            val_labels = val_labels_singlephase
        if test_labels is None:
            test_labels = test_labels_singlephase
    print 'before execute_classify, the shape of train_features is ', np.shape(
        train_features), ', the shape of val_features is ', np.shape(
        val_features), ', the shape of test_featrues is ', np.shape(test_features)
    acc = execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels)
    print acc
    return train_features, train_labels, val_features, val_labels, test_features, test_labels


if __name__ == '__main__':
    generate_representor_multidir_multiphase(
        data_dir='/home/give/Documents/dataset/MICCAI2018/Slices/crossvalidation/0',
        dictionary_dir='/home/give/PycharmProjects/MICCAI2018/LeaningBased/BoVW/dictionary'
    )

