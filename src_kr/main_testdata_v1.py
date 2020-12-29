import os
import datetime
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.pyplot import MultipleLocator
import os
import sys
import wfdb
import pywt
import seaborn
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA
import heartpy as hp


# 项目目录
# project_path = "D:\\python\\mit-bih_ecg_recognition\\"
project_path = "C:\\Users\\azuna\\PycharmProjects\\pythonProject\\ECG_project\\mit-bih_ecg_recognition-master\\mit-bih_ecg_recognition-master\\"
# 定义日志目录,必须是启动web应用时指定目录的子目录,建议使用日期时间作为子目录名
log_dir = project_path + "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model.h5"

# 测试集在数据集中所占的比例
RATIO = 1.0

def plot_data_single(data):
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
    set_plot_config(x_len=600)
        # print(data[i - 1].shape[0])  # shape is (625 * 12)
    plt.plot(np.array(range(data.shape[0])), data.T)
    plt.show()


def plot_data(data):
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

    for i in range(1, 13):
        plt.subplot(4, 3, i)
        set_plot_config(x_len=600)
        # print(data[i - 1].shape[0])  # shape is (625 * 12)
        plt.plot(np.array(range(data.shape[0])), data.T[i - 1])
    plt.show()


# 小波去噪预处理
def denoise(data):
    # 小波变换
    # plot_data_single(data)
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    # plot_data_single(rdata)
    return rdata[:-1]


# 读取心电数据和对应标签,并对数据进行小波去噪
def my_getDataSet(file_name, X_data, Y_data):
    label_map = {'正常心电图': 0, '窦性心动过缓': 1,
                 '窦性心动过速': 2, '窦性心律_不完全性右束支传导阻滞': 3,
                 '窦性心律_完全性右束支传导阻滞': 4, '窦性心律_完全性左束支传导阻滞': 5,
                 '窦性心律_左前分支阻滞': 6, '窦性心律_左室肥厚伴劳损': 7,
                 '窦性心律_提示左室肥厚': 8, '窦性心律_电轴左偏': 9}
    #
    # label_map = {'窦性心律_左室肥厚伴劳损': 0,
    #  '窦性心动过缓': 1, '窦性心动过速': 2,
    #  '窦性心律_不完全性右束支传导阻滞': 3,
    #  '窦性心律_电轴左偏': 4,
    #  '窦性心律_提示左室肥厚': 5,
    #  '窦性心律_完全性右束支传导阻滞': 6,
    #  '窦性心律_完全性左束支传导阻滞': 7,
    #  '窦性心律_左前分支阻滞': 8,
    #  '正常心电图': 9}


    def train_process(file_name="65593.mat", show=False):
        # print(m.keys())
        # print(m['Beats'])
        data = loadmat(file_name)['Beats']

        # draw_test_data(data.T)

        label = data['label'][0][0][0]
        # beatNumber = data['beatNumber'][0]
        rPeak = data['rPeak'][0][0][0]
        # tmp1 = data['beatData'][0]
        # tmp2 = data['beatData'][0][0]

        # tmp3 = data['beatData'][0][0][0]  # 22个心跳 22*N*12
        tmp4 = data['beatData'][0][0][0][0]  # 一个心跳 N*12

        # after_reshape = tmp3.reshape(-1, 1)

        # c = np.r_[data['beatData'][0][0][0][0], data['beatData'][0][0][0][1]]

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

        label = [label_map[label]]*len(rPeak)
        R_location = []
        for i in range(len(rPeak)):
            R_location.append(np.sum(rPeak[0:i + 1]) - 1)

        if show:

            for i in range(0, 12):
                plt.subplot(4, 3, i + 1)
                set_plot_config(x_len=beatData[i - 1].shape[0])
                print(beatData[i - 1].shape[0])  # shape is (625 * 12)
                plt.plot(np.array(range(beatData[i - 1].shape[0])), beatData.T[i - 1])
            plt.show()
        return tmp4, label, R_location

    def process_one_test_file(filename="89592.mat"):
        m = loadmat(filename)
        data = m['data'] # 10000*13
        data = data.T[1:] # 13 * 10000
        return data.T

    def find_rPeak(data):
        one_line = data.T[1]
        plot_data_single(one_line)
        one_line = hp.scale_data(one_line)


        # 自动获取R波位置
        working_data, measures = hp.process(one_line, 500.0)
        return working_data['peaklist']

    data = process_one_test_file(file_name)

    # data, Rclass, Rlocation = train_process(file_name)


    rdata = []
    for i in range(0, 12):
        rdata.append(denoise(data=data.T[i]))

    rdata = np.array(rdata) # 12*10000

    Rlocation = find_rPeak(rdata.T)

    # 去掉前后的不稳定数据
    start = 0
    end = 0
    i = start
    j = len(Rlocation) - end

    rdata = rdata.T  #  10000 * 12
    while i < j:
        try:
            lable = 9
            x_train = rdata[Rlocation[i] - 99 :Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


def file_name(file_dir=os.path.dirname(__file__)):
    path = os.walk(r'' + file_dir + r'\\UnlabeledData')

    file_names = []
    for _, _, file_2 in path:
        # print(len(file_2))
        # print(dir_names[index])

        for name in file_2:
            file_names.append(r'' + file_dir + r'\\UnlabeledData' + r'\\' + name)

    return file_names, file_2


# 加载数据集并进行预处理
def loadData(name):


    dataSet = []
    lableSet = []
    # n = r'C:\Users\azuna\PycharmProjects\pythonProject\ECG_project\src_kr\UnlabeledData\88247.mat'
    # for n in all_train_data_name:
    #     print(n)
    my_getDataSet(name, dataSet, lableSet)

    # print(all_train_data_name[100])
    # my_getDataSet(all_train_data_name[100], dataSet, lableSet)

    # 转numpy数组,打乱顺序
    newDataSet = dataSet[1][np.newaxis,:]
    newLabelSet = [lableSet[0]]
    for i in range(len(dataSet[1:])):
    # for d in dataSet[1:]:
        if dataSet[i].shape[0] == 300:
        # print(d.shape)
        #     print(i)
            newDataSet = np.concatenate((newDataSet, dataSet[i][np.newaxis,:]))
            newLabelSet.append(lableSet[i])


    lableSet = np.array(lableSet).reshape(-1, 1)

    # train_ds = np.hstack((dataSet, lableSet))
    newLabelSet = np.array(newLabelSet)
    # train_ds = np.hstack((newDataSet, newLabelSet))

    # np.random.shuffle(train_ds)

    # 数据集及其标签集
    # X = train_ds[:, :300].reshape(-1, 300, 1)
    # Y = train_ds[:, 300]
    X = newDataSet
    Y = newLabelSet

    # 测试集及其标签集
    # shuffle_index = np.random.permutation(len(X))
    # test_length = int(RATIO * len(shuffle_index))
    # test_index = shuffle_index[:test_length]
    # train_index = shuffle_index[test_length:]
    # X_test, Y_test = X[test_index], Y[test_index]
    # X_train, Y_train = X[train_index], Y[train_index]
    return X, Y


# 混淆矩阵
def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_pred, Y_pred)
    # 归一化
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # 绘图
    plt.figure(figsize=(10, 10))
    seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def main():

    # correct_label = {'正常心电图': 0, '窦性心动过缓': 1,
    #              '窦性心动过速': 2, '窦性心律_不完全性右束支传导阻滞': 3,
    #              '窦性心律_完全性右束支传导阻滞': 4, '窦性心律_完全性左束支传导阻滞': 5,
    #              '窦性心律_左前分支阻滞': 6, '窦性心律_左室肥厚伴劳损': 7,
    #              '窦性心律_提示左室肥厚': 8, '窦性心律_电轴左偏': 9}

    correct_label = {0:9, 1:1, 2:2, 3:3, 4:6, 5:7, 6:8,7:0 ,8:5, 9:4}

    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    all_test_data_name, pre_name = file_name()

    pred_map = {}
    i = 0
    for n in all_test_data_name:
        # try:
        X_test, Y_test = loadData(n)

        model = tf.keras.models.load_model(filepath=model_path)
        # 预测
        Y_pred = model.predict_classes(X_test)
        # 绘制混淆矩阵
        # plotHeatMap(Y_pred, Y_pred)
        # print(n)
        print(pre_name[i], end=' ')

        count = np.bincount(Y_pred)  # 找出第3列最频繁出现的值
        value = np.argmax(count)

        pred_map[pre_name[i][:-4]] = correct_label[value]
        print(correct_label[value])
        i+=1
        print(i)
        # except:
        #     pass



    ordered_name = []
    import csv

    with open('data.csv', 'r') as f:
        reader = csv.reader(f)
        # print(type(reader))
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            # print(row)
            ordered_name.append(row[0])


    # 1. 创建文件对象
    f = open('final.csv', 'w', encoding='utf-8', newline="")

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(["id", "categories"])
    for n in ordered_name:
        csv_writer.writerow([n, pred_map[n]])

    # 5. 关闭文件
    f.close()
if __name__ == '__main__':
    main()
