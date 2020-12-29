from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import MultipleLocator
import os
import sys
import heartpy as hp

label_map = {'正常心电图':0, '窦性心动过缓':1,
             '窦性心动过速':2, '窦性心律_不完全性右束支传导阻滞':3,
             '窦性心律_完全性右束支传导阻滞':4, '窦性心律_完全性左束支传导阻滞':5,
             '窦性心律_左前分支阻滞':6, '窦性心律_左室肥厚伴劳损':7,
             '窦性心律_提示左室肥厚':8, '窦性心律_电轴左偏':9}


class SlidingWindowAveragerator(object):

    # YOUR CODE HERE
    def __init__(self, window_size):
        self.window_size = window_size
        self.seq = []

    def add(self, x):
        if len(self.seq) == self.window_size:
            self.seq.pop(0)
        self.seq.append(x)

    @property
    def avg(self):
        return np.average(self.seq)

    @property
    def std(self):
        return np.std(self.seq)


def process_one_test_file(filename="89592.mat"):
    m = loadmat(filename)
    data = m['data']
    data = data.T

    one_line = data[1]
    one_line = hp.scale_data(one_line)

    # 自动获取R波位置
    working_data, measures = hp.process(one_line, 500.0)

    filter = SlidingWindowAveragerator(10)

    smoothed_data_list = []

    for line in data[1:]:
        new_data = np.array([])
        for i in line:
            filter.add(i)
            new_data = np.append(new_data, filter.avg)
        smoothed_data_list.append(new_data)

    smoothed_data_ndarray = np.array([arr for arr in smoothed_data_list])

    def set_plot_config():
        # 坐标轴刻度间隔设置
        x_major_locator = MultipleLocator(50)
        y_major_locator = MultipleLocator(0.5)

        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        # 坐标轴总显示范围设置
        plt.xlim(-0.5, 350)
        plt.ylim(-1.5, 1.5)

    for i in range(1, 13):
        plt.subplot(4, 3, i)
        set_plot_config()
        plt.plot(data[0], data[i].T)
    plt.show()

    for i in range(1, 13):
        plt.subplot(4, 3, i)
        set_plot_config()
        plt.plot(data[0], smoothed_data_ndarray[i - 1].T)
    plt.show()

    # plt.subplot(2, 1, 1)
    # set_plot_config()
    # plt.plot(data[0], data[1:].T)
    # print(data[0].shape, data[1:].shape)
    #
    # plt.subplot(2, 1, 2)
    # set_plot_config()
    # plt.plot(data[0], smoothed_data_ndarray.T)
    # print(data[0].shape, smoothed_data_ndarray.shape)

    plt.show()


def train_process(file_name="65593.mat", show=False):
    # print(m.keys())
    # print(m['Beats'])
    data = loadmat(file_name)['Beats']

    # draw_test_data(data.T)

    label = data['label'][0][0][0]
    beatNumber = data['beatNumber'][0]
    rPeak = data['rPeak'][0][0][0]
    tmp1 = data['beatData'][0]
    tmp2 = data['beatData'][0][0]

    tmp3 = data['beatData'][0][0][0]  # 22个心跳 22*N*12
    tmp4 = data['beatData'][0][0][0][0]  # 一个心跳 N*12

    after_reshape = tmp3.reshape(-1,1)

    c = np.r_[data['beatData'][0][0][0][0], data['beatData'][0][0][0][1]]

    beatData = data['beatData'][0][0][0]

    for i in range(1, len(rPeak)):
        tmp4 = np.insert(tmp4, 0, values=data['beatData'][0][0][0][i], axis=0)
        # c = np.insert(a, 0, values=b, axis=0)
    def set_plot_config(x_len=300):
        # 坐标轴刻度间隔设置
        x_major_locator = MultipleLocator(200)
        y_major_locator = MultipleLocator(0.5)

        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        # 坐标轴总显示范围设置
        plt.xlim(-0.5, x_len)
        plt.ylim(-1.5, 1.5)

    # for i in range(1, 13):
    #     plt.subplot(4, 3, i)
    #     set_plot_config()
    #     plt.plot(data[0], data[i].T)
    # plt.show()


    label = label_map[label]
    R_location = []
    for i in range(len(rPeak)):
        R_location.append(np.sum(rPeak[0:i+1])-1)

    if show:

        for i in range(0, 12):
            plt.subplot(4, 3, i+1)
            set_plot_config(x_len = beatData[i - 1].shape[0])
            print(beatData[i - 1].shape[0])  # shape is (625 * 12)
            plt.plot(np.array(range(beatData[i - 1].shape[0])), beatData.T[i - 1])
        plt.show()


def file_name(file_dir=os.path.dirname(__file__)):
    path = os.walk(r'' + file_dir + r'\\Train')

    dir_names = []
    for _, dir_name, _ in path:
        dir_names.extend(dir_name)
        break
    print(dir_names)

    file_names = []
    index = 0
    for _, _, file_2 in path:
        # print(len(file_2))
        # print(dir_names[index])

        for name in file_2:
            file_names.append(r'' + file_dir + r'\\Train\\' + dir_names[index] + r'\\' + name)
        index += 1

    return file_names


def main():
    # all_train_data_name = file_name()

    # train_process(all_train_data_name[0])
    process_one_test_file()

if __name__ == '__main__':
    main()
