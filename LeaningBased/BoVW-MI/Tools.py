import numpy as np


class Tools:
    @staticmethod
    def multithresh(array, levels):
        res = []
        min_value = np.min(array)
        max_value = np.max(array)
        pre_split_value = (1.0 * (max_value - min_value)) / (1.0 * levels)
        start_value = min_value
        for i in range(levels):
            res.append(start_value)
            start_value += pre_split_value
        return res

    @staticmethod
    def imquantize(array, thresh):
        for index, item in enumerate(array):
            for i in range(len(thresh)):
                if item >= thresh[i]:
                    array[index] = i
        return array

if __name__ == '__main__':
    thresh = Tools.multithresh(range(10), 5)
    quantize_arr = Tools.imquantize(range(10), thresh)
    print thresh
    print range(10)
    print quantize_arr