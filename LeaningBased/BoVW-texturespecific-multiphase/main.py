import os
import numpy as np
from ExtractPatches import extract_patches_multidir
from utils.Tools import calculate_acc_error, check_save_path
import scipy.io as scio

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
    names = os.listdir(data_dir)
    vocabulary_dict = {}
    for name in names:
        vocabulary_path = os.path.join(data_dir, name, 'vocabulary.npy')
        vocabulary_dict[int(float(name))] = np.load(vocabulary_path)
    return vocabulary_dict

def generate_representor(data_dir, patch_dir, subclass):
    vocabulary_dict = load_vocabulary(patch_dir)
    shape_vocabulary = np.shape(vocabulary_dict[0])
    vocabulary_size = shape_vocabulary[0]
    representers = []
    patches, coding_labeles, labeles = extract_patches_multidir(data_dir, subclasses=[subclass], return_flag=True)
    for case_index, cur_patches in enumerate(patches):
        cur_case_representor = np.zeros([10, vocabulary_size])
        patches_coding_labeles = {}
        for patch_index, cur_patch in enumerate(cur_patches):
            cur_coding_label = coding_labeles[case_index][patch_index]
            if cur_coding_label not in patches_coding_labeles.keys():
                patches_coding_labeles[cur_coding_label] = []
            patches_coding_labeles[cur_coding_label].append(cur_patch)
        for key in patches_coding_labeles.keys():
            cur_patches_coding_label = patches_coding_labeles[key]
            cur_vocabulary = vocabulary_dict[key]
            distance_arr = cal_distance(cur_patches_coding_label, cur_vocabulary)
            for i in range(len(distance_arr)):
                min_index = np.argmin(distance_arr[i])
                cur_case_representor[int(key), min_index] += 1
        representers.append(cur_case_representor.flatten())
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

def generate_representor_multidir(data_dir, patch_dir, reload=None, save_dir=None):
    import scipy.io as scio
    if reload is None:
        train_features, train_labels = generate_representor(data_dir, patch_dir=patch_dir,
                                                            subclass='train')
        check_save_path(os.path.join(save_dir, 'training.mat'))
        scio.savemat(os.path.join(save_dir, 'training.mat'), {
            'features': train_features,
            'labels': train_labels
        })
        test_features, test_labels = generate_representor(data_dir, patch_dir=patch_dir,
                                                          subclass='test')
        scio.savemat(os.path.join(save_dir, 'testing.mat'), {
            'features': test_features,
            'labels': test_labels
        })
        val_features, val_labels = generate_representor(data_dir, patch_dir=patch_dir, subclass='val')
        scio.savemat(os.path.join(save_dir, 'validation.mat'), {
            'features': val_features,
            'labels': val_labels
        })
    else:
        train_data = scio.loadmat(os.path.join(save_dir, 'training.mat'))
        train_features = train_data['features']
        train_labels = train_data['labels']

        test_data = scio.loadmat(os.path.join(save_dir, 'testing.mat'))
        test_features = test_data['features']
        test_labels = test_data['labels']

        val_data = scio.loadmat(os.path.join(save_dir, 'validation.mat'))
        val_features = val_data['features']
        val_labels = val_data['labels']
    acc = execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels)
    return acc
if __name__ == '__main__':

    '''
    iter_num = 10
    from GenerateDictionary import generate_dictionary_multidir
    generate_dictionary_multidir(
        '/home/give/Documents/dataset/MICCAI2018/Patches/LearningBased/BoVW-TextureSpecific-multiphase')
    record_file = './record.txt'
    filed = open(record_file, 'w')
    for iter_index in range(iter_num):
        accs = []
        for crossid in [0, 1]:
            acc = generate_representor_multidir(
                patch_dir='/home/give/Documents/dataset/MICCAI2018/Patches/LearningBased/BoVW-TextureSpecific-multiphase',
                data_dir='/home/give/Documents/dataset/MICCAI2018/Slices/un-crossvalidation',
                save_dir=os.path.join(
                    '/home/give/PycharmProjects/MICCAI2018/LeaningBased/BoVW-texturespecific-multiphase/findbest',
                    str(crossid)))
            accs.append(acc)
        print str(iter_index) + ': ' + str(accs[0]) + ' , ' + str(accs[1]) + '\n'
        filed.write(str(iter_index) + ': ' + str(accs[0]) + ' , ' + str(accs[1]) + '\n')
    filed.close()
    '''
    crossid = 1
    acc = generate_representor_multidir(
        patch_dir='/home/give/Documents/dataset/MICCAI2018/Patches/LearningBased/BoVW-TextureSpecific-multiphase',
        data_dir='/home/give/Documents/dataset/MICCAI2018/Slices/un-crossvalidation',
        save_dir=os.path.join(
            '/home/give/PycharmProjects/MICCAI2018/LeaningBased/BoVW-texturespecific-multiphase/findbest',
            str(crossid)))