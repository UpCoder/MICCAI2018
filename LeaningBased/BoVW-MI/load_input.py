from sklearn.decomposition import PCA
import scipy.io as scio
import numpy as np


def load_input():
    # load label
    labels_path = './data/ids.mat'
    labels_file = scio.loadmat(labels_path)['ids']
    labels = []
    for index, label in enumerate(labels_file):
        labels.append(int(label[3]))
    print 'load label finish,  label shape is ', np.shape(labels)
    # load patch
    patchs_path = './data/SegPatches.mat'
    data = scio.loadmat(patchs_path)['patches']
    patches = []
    count = []  # record  the number of patches of pre image
    for item in data:
        temp_arr = list(
            np.array(
                np.matrix(item[0]).T
            )
        )
        count.append(len(temp_arr))
        patches.extend(
            temp_arr
        )
    print 'load count finished, count sum is ', np.sum(count)
    print 'load patches finish, patches shape is ', np.shape(patches)
    return labels, patches, count

if __name__ == '__main__':
    load_input()