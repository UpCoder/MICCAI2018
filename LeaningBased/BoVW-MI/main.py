# wang jian version
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import load_input
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


if __name__ == '__main__':
    labels, patches, count = load_input.load_input()
    # labels, patches, count = get_small_patch(labels, np.array(patches), count)
    patches = do_pca(patches)
    print 'finish pca operation, the patches shape is ', np.shape(patches)
    kmeans_res = do_k_means(patches)
    images_hist = calu_hist(count, kmeans_res.labels_)
    # SVM.do_svm(np.array(images_hist), np.array(labels))
    print 'image_hist shape is ', np.shape(images_hist)
    images_hist_quantize = do_quantize(images_hist)
    MI = calu_MI(images_hist_quantize, labels)
    selected_columns = select_column(MI, MI_SELECT_NUM)
    images_hist_select = images_hist[:, selected_columns]
    print 'image select shape is ', np.shape(images_hist_select)
    SVM.do_svm(np.array(images_hist_select), np.array(labels))
