from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import MultipleLocator


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



def draw_test_data():
    m = loadmat("89592.mat")
    data = m['data']
    data = data.T



    smoothx = SlidingWindowAveragerator(10)

    smooth_arr = []

    for line in data[1:].T:
        new_data = np.array([])
        for i in line:
            smoothx.add(i)
            new_data = np.append(new_data, smoothx.avg)
        smooth_arr.append(new_data)

    new_many = np.array([arr for arr in smooth_arr])

    plt.subplot(2, 1, 1)
    x_major_locator = MultipleLocator(50)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(-0.5, 350)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-1.5, 1.5)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.plot(data[0], data[1:].T)


    plt.subplot(2, 1, 2)
    x_major_locator = MultipleLocator(50)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.5)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(-0.5, 350)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(-1.5, 1.5)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.plot(data[0], new_many)
    print(len(data[0]), len(data[2]))
    # plt.plot([1,2,3], [5,6,1])
    plt.show()


draw_test_data()

def draw_train_data(data):

    x_label = np.array(range(len(data)))
    plt.plot(x_label, data.T[0])
    print(len(data))
    # plt.plot([1,2,3], [5,6,1])
    plt.show()



def train_process():

    m = loadmat("65593.mat")
    # m = loadmat("89592.mat")

    # print(m.keys())
    # print(m['Beats'])
    data = m['Beats']


    # draw_test_data(data.T)

    a0 = data[0]

    # label = a0['label'][0]
    # beatNumber = a0['beatNumber'][0]
    # rPeak = a0['rPeak'][0]
    # print(len(a0['beatData'][0][0][0]))
    # miaomiaomiao = a0['beatData'][0][0]
    beatData = a0['beatData'][0][0][0]
    draw_train_data(beatData)

    # scipy.io.whosmat("89592.mat")

print('')


