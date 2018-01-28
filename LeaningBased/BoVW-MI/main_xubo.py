# -*- coding=utf-8 -*-
# xu bo version
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from glob import glob
from utils.Tools import read_mhd_image, get_boundingbox, image_erode, convert2depthlaster, calculate_acc_error
from LeaningBased.selected_patches import flatten_arr
import Tools
import math
import SVM

KMEANS_CENTER_NUM = 300
MI_SELECT_NUM = 100
# do pca operation for X(patches)
# column_num is the param that determine the number of dimension after doing pca operation
def do_pca(X, column_num=10):
    pca = PCA(n_components=column_num)
    X = pca.fit_transform(X)
    return X


# calu hist
# count is the number of patches each image
# categories is pre patch's category
def calu_hist(count, categories, k_num=KMEANS_CENTER_NUM):
    hist = []
    start_index = 0
    for num in count:
        pre_hist = [0] * k_num
        for _ in range(num):
            pre_hist[categories[start_index]] += 1
            start_index += 1
        hist.append(pre_hist)
    return np.array(hist)


# will quantize the ith feature to L levels
def do_quantize(hist, levels=5):
    columns_num = np.shape(hist)[1]
    hist_out = np.zeros(np.shape(hist))
    for i in range(columns_num):
        one_column = hist[:, i]
        thresh = Tools.Tools.multithresh(one_column, levels)
        hist_out[:, i] = Tools.Tools.imquantize(one_column, thresh)
    return hist_out

def do_k_means(X, k_num=KMEANS_CENTER_NUM):
    kmeans_res = KMeans(n_clusters=k_num, random_state=0, n_jobs=-1).fit(X)
    return kmeans_res


# calu p(v, c)
# count the number of sample that label is c and ith's feature is v
def calu_p_v_c(hist, labels, i, v, c):
    t = 0
    for index, item in enumerate(hist[:, i]):
        if labels[index] == c and item == v:
            t += 1
    return (1.0 * t) / (1.0 * len(hist[:, i]))

# calu p(v)
def calu_p_v(hist, i, v):
    t = 0
    for item in hist[:, i]:
        if item == v:
            t += 1
    return (1.0 * t) / (1.0 * len(hist[:, i]))
# calu p(c)
def calu_p_c(labels, c):
    t = 0
    for item in labels:
        if item == c:
            t += 1
    return (1.0 * t) / (1.0 * len(labels))

# calu mutual information
# calu pre feature(word)'s importance
def calu_MI(images_hist, labels):
    columns_num = np.shape(images_hist)[1]
    MI = np.zeros(columns_num, dtype=np.float32)
    for i in range(columns_num):
        # for pre feature:
        min_level = 0
        max_level = 5
        one_mi = 0.0
        for level in range(min_level, max_level):
            for label in list(set(labels)):
                pvc = calu_p_v_c(images_hist, labels, i, level, label)
                pv = calu_p_v(images_hist, i, level)
                pc = calu_p_c(labels, label)
                if pv != 0 and pc != 0 and pvc != 0:
                    one_mi += (pvc * 1.0) * math.log(pvc / (pv * pc))
        MI[i] = one_mi
    return MI


# select num indexs by MI
def select_column(MI, num):
    sorted_MI = sorted(MI, reverse=True)
    res = []
    flag = np.zeros(len(MI))
    for i in range(num):
        for index in range(len(MI)):
            if MI[index] == sorted_MI[i] and flag[index] == 0:
                res.append(index)
                flag[index] = 1
                break

    return res


def calu_all_labels(labels, count):
    res = []
    for index, item in enumerate(labels):
        res.extend([item] * count[index])
    return res

def get_small_patch(labels, patches, count):
    selected_num = 20
    select_indexs = np.random.randint(0, len(count), selected_num)
    new_labels = []
    new_patches = []
    new_count = []
    category_count = np.zeros(5)
    for index in select_indexs:
        new_labels.append(labels[index])
        category_count[labels[index]-1] += 1
        new_count.append(count[index])
        start_index = np.sum(count[:index])
        end_index = np.sum(count[:index+1])
        new_patches.extend(patches[start_index:end_index, :])
    print 'category distribution is ', category_count
    return new_labels, new_patches, new_count


phasenames = ['NC', 'ART', 'PV']
def extract_interior_patch_npy(dir_name, suffix_name, save_dir, patch_size, patch_step=1, erode_size=1):
    '''
    提取指定类型病灶的ｐａｔｃｈ 保存原始像素值，存成ｎｐｙ的格式
    :param patch_size: 提取ｐａｔｃｈ的大小
    :param dir_name: 目前所有病例的存储路径
    :param suffix_name: 指定的病灶类型的后缀，比如说cyst 就是０
    :param save_dir:　提取得到的ｐａｔｃｈ的存储路径
    :param patch_step: 提取ｐａｔｃｈ的步长
    :param erode_size: 向内缩的距离，因为我们需要确定内部区域,所以为了得到内部区域，我们就将原区域向内缩以得到内部区域
    :return: None
    '''
    count = 0
    names = os.listdir(dir_name)
    patches = []
    for name in names:
        if name.endswith(suffix_name):
            # 只提取指定类型病灶的ｐａｔｃｈ
            mask_images = []
            mhd_images = []
            flag = True
            for phasename in phasenames:
                image_path = glob(os.path.join(dir_name, name, phasename + '_Image*.mhd'))[0]
                mask_path = os.path.join(dir_name, name, phasename + '_Registration.mhd')
                mhd_image = read_mhd_image(image_path)
                mhd_image = np.squeeze(mhd_image)
                # show_image(mhd_image)
                mask_image = read_mhd_image(mask_path)
                mask_image = np.squeeze(mask_image)

                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
                    flag = False
                    continue
                mask_image = image_erode(mask_image, erode_size)
                [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
                mask_image = mask_image[xmin: xmax, ymin: ymax]
                mhd_image = mhd_image[xmin: xmax, ymin: ymax]
                # mhd_image[mask_image != 1] = 0
                mask_images.append(mask_image)
                mhd_images.append(mhd_image)
                # show_image(mhd_image)
            if not flag:
                continue
            mask_images = convert2depthlaster(mask_images)
            mhd_images = convert2depthlaster(mhd_images)
            count += 1
            [width, height, depth] = list(np.shape(mhd_images))
            patch_count = 1
            cur_patches = []
            # if width * height >= 900:
            #     patch_step = int(math.sqrt(width * height / 100))
            for i in range(patch_size / 2, width - patch_size / 2, patch_step):
                for j in range(patch_size / 2, height - patch_size / 2, patch_step):
                    cur_patch = mhd_images[i - patch_size / 2:i + patch_size / 2,
                                j - patch_size / 2: j + patch_size / 2, :]
                    if (np.sum(mask_images[i - patch_size / 2:i + patch_size / 2,
                               j - patch_size / 2: j + patch_size / 2, :]) / (
                                    (patch_size - 1) * (patch_size - 1) * 3)) < 0.5:
                        continue
                    if save_dir is not None:
                        save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.npy')
                        np.save(save_path, np.array(cur_patch))
                    else:
                        cur_patches.append(flatten_arr(np.asarray(cur_patch)))
                    patch_count += 1

            if save_dir is None:
                if len(cur_patches) == 0:
                    continue
                patches.append(cur_patches)
            if patch_count == 1:
                continue
                # save_path = os.path.join(save_dir, name + '_' + str(patch_count) + '.png')
                # roi_image = Image.fromarray(np.asarray(mhd_images, np.uint8))
                # roi_image.save(save_path)
    print count
    if save_dir is None:
        return patches


def generate_train_val_features(interior_cluster_centriod_path, extract_patch_function):
    train_interior_patches = []
    train_labels = []
    category_number = [0, 1, 2, 3]
    patch_size = 7
    for i in category_number:
        interior_patches = extract_patch_function(
                '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/train',
                str(i),
                save_dir=None,
                patch_size=(patch_size + 1)
            )
        train_labels.extend([i] * len(interior_patches))
        train_interior_patches.extend(interior_patches)
    train_interior_patches = np.array(train_interior_patches)
    print 'train_interior_patches shape is ', np.shape(train_interior_patches)

    val_interior_patches = []
    val_labels = []
    for i in category_number:
        interior_patches = extract_patch_function(
            '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/ICIP/test',
            str(i),
            save_dir=None,
            patch_size=(patch_size + 1),
        )
        val_labels.extend([i] * len(interior_patches))
        val_interior_patches.extend(interior_patches)
    val_interior_patches = np.array(val_interior_patches)
    print 'train_interior_patches shape is ', np.shape(val_interior_patches)

    interior_cluster_centroid_arr = np.load(interior_cluster_centriod_path)
    train_interior_features = []
    for i in range(len(train_interior_patches)):
        train_interior_features.append(
            generate_patches_representer(train_interior_patches[i], interior_cluster_centroid_arr).squeeze()
        )
    val_interior_features = []
    for i in range(len(val_interior_patches)):
        val_interior_features.append(
            generate_patches_representer(val_interior_patches[i], interior_cluster_centroid_arr).squeeze()
        )
    print 'the shape of train interior features is ', np.shape(train_interior_features)
    print 'the shape of val interior features is ', np.shape(val_interior_features)
    # scio.savemat(
    #     './data_128_False.mat',
    #     {
    #         'train_features': train_features,
    #         'train_labels': train_labels,
    #         'val_features': val_features,
    #         'val_labels': val_labels
    #     }
    # )
    return train_interior_features, train_labels, \
           val_interior_features, val_labels


def generate_patches_representer(patches, cluster_centers):
    '''
    用词典表示一组patches
    :param patches: 表示一组patch　（None, 192）
    :param cluster_centers:　(vocabulary_size, 192)
    :return: (1, vocabulary_size) 行向量　表示这幅图像
    '''
    print np.shape(patches)
    print np.shape(cluster_centers)
    shape = list(np.shape(cluster_centers))
    mat_cluster_centers = np.mat(cluster_centers)
    mat_patches = np.mat(patches)
    mat_distance = mat_patches * mat_cluster_centers.T # (None, vocabulary_size)
    represented_vector = np.zeros([1, shape[0]])
    for i in range(len(mat_distance)):
        distance_vector = np.array(mat_distance[i])
        min_index = np.argmin(distance_vector, axis=1)
        represented_vector[0, min_index] += 1
    return represented_vector


def generate_dictionary(patch_dir, cluster_save_path, target_labels=[0, 1, 2, 3], cluster_num=256, pre_class_num=30000):
    from LeaningBased.selected_patches import return_patches_multidir
    features = return_patches_multidir(
        patch_dir,
        subclass_names=['train', 'test'],
        target_label=target_labels,
        pre_class_num=pre_class_num
    )
    print 'before execute do_kmeans, the shape of features is ', np.shape(features)
    vocabulary = do_kmeans(features, vocabulary_size=cluster_num)
    np.save(cluster_save_path, vocabulary)


def do_kmeans(fea, vocabulary_size=128):
    print 'fea shape is ', np.shape(fea), ' vocabulary size is ', vocabulary_size
    kmeans_obj = KMeans(n_clusters=vocabulary_size, n_jobs=8).fit(fea)
    cluster_centroid_objs = kmeans_obj.cluster_centers_
    print np.shape(cluster_centroid_objs)
    return cluster_centroid_objs


def execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    from LeaningBased.BoVW_DualDict.classification import SVM, LinearSVM, KNN
    predicted_label, c_params, g_params, max_c, max_g, accs = SVM.do(train_features, train_labels, val_features,
                                                                     val_labels,
                                                                     adjust_parameters=True)
    predicted_label, accs = SVM.do(train_features, train_labels, test_features, test_labels,
                                   adjust_parameters=False, C=max_c, gamma=max_g)
    calculate_acc_error(predicted_label, test_labels)
    print 'ACA is ', accs
    return accs

if __name__ == '__main__':
    '''
    import scipy.io as scio
    import os
    mat_path = '/home/give/PycharmProjects/BoVWMI/Code/BoVW-MI/data/xubo/histogram.mat'
    label_path = '/home/give/PycharmProjects/BoVWMI/Code/BoVW-MI/data/xubo/class.mat'
    images_hist = scio.loadmat(mat_path)['histogram']
    images_hist = np.array(images_hist)
    labels = scio.loadmat(label_path)['class']
    labels = np.reshape(labels, [132])
    print 'image_hist shape is ', np.shape(images_hist)
    print 'labels shape is ', np.shape(labels)
    # SVM.do_svm(np.array(54), np.array(labels))
    '''
    import scipy.io as scio
    import os
    from shutil import copy
    iterator_num = 1
    for iterator_index in range(0, iterator_num):
        print 'iterator_index is ', iterator_index
        # dictionary_path = './dictionarys/' + str(iterator_index) + '_dictionary.npy'
        # # 提取字典
        # generate_dictionary(
        #     '/home/give/Documents/dataset/ICPR2018/BoVW-MI/patches',
        #     dictionary_path,
        #     cluster_num=256,
        #     pre_class_num=30000,
        # )
        #
        # # 构造特征
        # train_interior_features, train_labels, \
        # val_interior_features, val_labels = generate_train_val_features(
        #     dictionary_path,
        #     extract_interior_patch_npy)
        # scio.savemat(
        #     './features.mat',
        #     {
        #         'train_features': train_interior_features,
        #         'train_labels': train_labels,
        #         'test_features': val_interior_features,
        #         'test_labels': val_labels
        #     }
        # )

        # 丢进分类器训练
        crossid = 1
        original_dir = '/home/give/PycharmProjects/MICCAI2018/LeaningBased/BoVW-DualDict-New/mat'
        save_dir = '/home/give/PycharmProjects/MICCAI2018/LeaningBased/BoVW-MI/mat'
        train_data = scio.loadmat(os.path.join(original_dir, str(crossid), 'training.mat'))
        train_features = train_data['features']
        train_num = len(train_features)
        train_labels = np.squeeze(train_data['labels'])

        test_data = scio.loadmat(os.path.join(original_dir, str(crossid), 'testing.mat'))
        test_features = test_data['features']
        test_num = len(test_features)
        test_labels = np.squeeze(test_data['labels'])

        val_data = scio.loadmat(os.path.join(original_dir, str(crossid), 'validation.mat'))
        val_features = val_data['features']
        val_num = len(val_features)
        val_labels = np.squeeze(val_data['labels'])

        # execute_classify(train_features, train_labels, test_features, test_labels)
        train_len = len(train_features)
        # val_len = len(val_features)
        images_hist = np.concatenate([train_features, val_features, test_features], axis=0)
        print np.shape(train_features), np.shape(test_features)
        print np.shape(train_labels), np.shape(test_labels)
        labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)
        MI_SELECT_NUMS = [150]
        for MI_SELECT_NUM in MI_SELECT_NUMS:
            # accuracy, param_record = SVM.do_svm(np.array(images_hist), np.array(labels), 3, False)
            # print 'MI SELECT NUM is %d, accuracy is %g' % (MI_SELECT_NUM, accuracy)

            images_hist_quantize = do_quantize(images_hist, levels=16)
            MI = calu_MI(images_hist_quantize, labels)
            selected_columns = select_column(MI, MI_SELECT_NUM)
            images_hist_select = images_hist[:, selected_columns]

            train_features = images_hist_select[: train_num, :]
            val_features = images_hist_select[train_num: train_num + val_num]
            test_features = images_hist_select[train_num + val_num: train_num + val_num + test_num]
            scio.savemat(os.path.join(save_dir, str(crossid), 'training.mat'), {
                'features': train_features,
                'labels': train_labels
            })
            scio.savemat(os.path.join(save_dir, str(crossid), 'validation.mat'), {
                'features': val_features,
                'labels': val_labels
            })
            scio.savemat(os.path.join(save_dir, str(crossid), 'testing.mat'), {
                'features': test_features,
                'labels': test_labels
            })
            acc = execute_classify(train_features, train_labels, val_features, val_labels, test_features, test_labels)
