from skimage.feature import local_binary_pattern
from utils.Tools import read_mhd_image, get_boundingbox, check_save_path, split_array
import os
from glob import glob
import numpy as np
from multiprocessing import Pool
import scipy.io as scio


def read_from_dir(data_dir, phase_name):
    mhd_path = glob(os.path.join(data_dir, phase_name + '_Image*.mhd'))[0]
    mhd_image = read_mhd_image(mhd_path)
    mask_path = os.path.join(data_dir, phase_name + '_Registration.mhd')
    mask_image = read_mhd_image(mask_path)
    mhd_image = np.squeeze(mhd_image)
    mask_image = np.squeeze(mask_image)
    return mask_image, mhd_image


def convert_coding(file_dir, phase_name):
    pv_mask_image, pv_mhd_image = read_from_dir(file_dir, phase_name=phase_name)
    [x_min, x_max, y_min, y_max] = get_boundingbox(pv_mask_image)
    roi_image = pv_mhd_image[x_min:x_max, y_min: y_max]
    after_conding = local_binary_pattern(roi_image, 8, 3, 'uniform')
    return after_conding


def extract_patches_multifiles(data_dir, names, target_label, patch_size, patch_step, save_dir, phase_name):
    patches = []
    labeles = []
    coding_labeles = []
    for name in names:
        if name is not None and not name.endswith(target_label):
            continue
        cur_data_dir = os.path.join(data_dir, name)
        print 'extract patches from ', cur_data_dir, ' at ', str(os.getpid())
        pv_mask_image, pv_mhd_image = read_from_dir(cur_data_dir, phase_name=phase_name)
        coding_image = convert_coding(cur_data_dir, phase_name=phase_name)
        [x_min, x_max, y_min, y_max] = get_boundingbox(pv_mask_image)
        r = patch_size / 2
        cur_patches = []
        cur_coding_labeles = []
        for i in range(x_min, x_max, patch_step):
            for j in range(y_min, y_max, patch_step):
                cur_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
                cur_mask_patch = pv_mhd_image[i - r: i + r + 1, j - r: j + r + 1]
                if ((1.0 * np.sum(cur_mask_patch)) / (1.0 * patch_size * patch_size)) < 0.1:
                    continue
                cur_label = target_label
                cur_coding_label = coding_image[i-x_min, j-y_min]

                if save_dir is not None:
                    save_path_dir = os.path.join(save_dir, str(cur_coding_label), str(cur_label))
                    if os.path.exists(save_path_dir):
                        cur_id = len(os.listdir(save_path_dir))
                    else:
                        cur_id = 0
                    save_path = os.path.join(save_dir, str(cur_coding_label), str(cur_label), str(cur_id) + '.npy')
                    check_save_path(save_path)
                    np.save(save_path, cur_patch)
                else:
                    cur_patches.append(np.array(cur_patch).flatten())
                    cur_coding_labeles.append(cur_coding_label)
        if save_dir is None:
            if len(cur_patches) == 0:
                continue
            patches.append(cur_patches)
            coding_labeles.append(cur_coding_labeles)
            labeles.append(int(target_label))
    print len(patches), len(coding_labeles), len(labeles)
    return patches, coding_labeles, labeles


def extract_patches_singledir(data_dir, target_label, patch_size, patch_step, phase_name, multiprocess=8):
    names = os.listdir(data_dir)
    patches = []
    labeles = []
    coding_labeles = []
    if multiprocess is None:
        patches, coding_labeles, labeles = extract_patches_multifiles(data_dir, names, target_label, patch_size,
                                                                      patch_step, None, phase_name=phase_name)
    else:
        names_group = split_array(names, multiprocess)
        pool = Pool()
        results = []
        for i in range(multiprocess):
            result = pool.apply_async(extract_patches_multifiles,
                                      (data_dir, names_group[i], target_label, patch_size, patch_step, None, phase_name, ))
            results.append(result)
        pool.close()
        pool.join()
        for i in range(multiprocess):
            cur_patches, cur_coding_labeles, cur_labeles = results[i].get()
            patches.extend(cur_patches)
            labeles.extend(cur_labeles)
            coding_labeles.extend(cur_coding_labeles)

    return patches, coding_labeles, labeles


def extract_patches_multidir(data_dir, subclasses=['train', 'val', 'test'], target_labels=[0, 1, 2, 3],
                             patch_size=7, patch_step=1, phase_name='PV',
                             save_path='/home/give/Documents/dataset/ICPR2018/BoVW-SparseCoding/data.mat', return_flag=False):
    patches = []
    labeles = []
    coding_labeles = []
    for subclass in subclasses:
        for target_label in target_labels:
            cur_data_dir = os.path.join(data_dir, subclass)
            cur_patches, cur_coding_labels, cur_labeles = extract_patches_singledir(cur_data_dir, str(target_label),
                                                                                    patch_size=patch_size,
                                                                                    phase_name=phase_name,
                                                                                    patch_step=patch_step,
                                                                                    multiprocess=8)
            patches.extend(cur_patches)
            coding_labeles.extend(cur_coding_labels)
            labeles.extend(cur_labeles)
    if return_flag:
        return patches, coding_labeles, labeles
    if save_path is not None:
        scio.savemat(save_path, {
            'patches': patches,
            'labeles': labeles
        })
        save_dict = {}

        for i, cur_patches in enumerate(patches):
            cur_label = labeles[i]
            if str(cur_label) in save_dict.keys():
                save_dict[str(cur_label)].extend(cur_patches)
            else:
                save_dict[str(cur_label)] = []
        scio.savemat(save_path, save_dict)

if __name__ == '__main__':
    for phase_name in ['NC', 'ART', 'PV']:
        extract_patches_multidir('/home/give/Documents/dataset/MICCAI2018/Slices/un-crossvalidation',
                                 phase_name=phase_name,
                                 save_path='/home/give/Documents/dataset/MICCAI2018/Patches/LearningBased/BoVW-Yang/data_' + phase_name + '.mat')

    # data = scio.loadmat('/home/give/Documents/dataset/ICPR2018/BoVW-SparseCoding/data.mat')
    # for key in data.keys():
    #     print np.shape(data[key])