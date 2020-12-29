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

# 项目目录
# project_path = "D:\\python\\mit-bih_ecg_recognition\\"
project_path = "C:\\Users\\azuna\\PycharmProjects\\pythonProject\\ECG_project\\mit-bih_ecg_recognition-master\\mit-bih_ecg_recognition-master\\"
# 定义日志目录,必须是启动web应用时指定目录的子目录,建议使用日期时间作为子目录名
log_dir = project_path + "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model.h5"

# 测试集在数据集中所占的比例
RATIO = 0.9

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


# 没用到----------------
# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


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
        beatNumber = data['beatNumber'][0]
        rPeak = data['rPeak'][0][0][0]
        tmp1 = data['beatData'][0]
        tmp2 = data['beatData'][0][0]

        tmp3 = data['beatData'][0][0][0]  # 22个心跳 22*N*12
        tmp4 = data['beatData'][0][0][0][0]  # 一个心跳 N*12

        after_reshape = tmp3.reshape(-1, 1)

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


    data, Rclass, Rlocation = train_process(file_name)

    # 读取心电数据记录
    # print("正在读取 " + number + " 号心电数据...")
    # record = wfdb.rdrecord('ecg_data/' + number, channel_names=['MLII'])
    # data = record.p_signal.flatten()
    # plot_data(data)
    # rdata = np.array([])
    # plot_data_single(data.T[0])
    rdata = []
    for i in range(0, 12):
        # remain = data.T[i]
        # rdata = np.append(rdata, denoise(data=data.T[i]))
        rdata.append(denoise(data=data.T[i]))
        # rdata = np.insert(rdata, 0, values=[denoise(data=data.T[i])], axis=0)
        # data.T[i] = denoise(data=data.T[i])
        # print(np.sum(denoise(data=data.T[i])-remain))
    rdata = np.array(rdata)
    # plot_data(rdata.T)


    # 去掉前后的不稳定数据
    start = 0
    end = 0
    i = start
    j = len(Rclass) - end

    rdata = rdata.T  #  10000 * 12
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    # ica = FastICA(n_components=10)
    while i < j:
        try:
            lable = Rclass[i]
            x_train = rdata[Rlocation[i] - 99 :Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


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


# 加载数据集并进行预处理
def loadData():
    all_train_data_name = file_name()

    dataSet = []
    lableSet = []

    for n in all_train_data_name:
        print(n)
        my_getDataSet(n, dataSet, lableSet)

    # print(all_train_data_name[100])
    # my_getDataSet(all_train_data_name[100], dataSet, lableSet)

    # 转numpy数组,打乱顺序
    newDataSet = dataSet[0][np.newaxis,:]
    newLabelSet = [lableSet[0]]
    for i in range(len(dataSet[1:])):
    # for d in dataSet[1:]:
        if dataSet[i].shape[0] == 300:
        # print(d.shape)
            print(i)
            newDataSet = np.concatenate((newDataSet, dataSet[i][np.newaxis,:]))
            newLabelSet.append(lableSet[i])

    # dataSet = np.array(dataSet).reshape((-1,300,12))
    # newDataSet = dataSet[0]
    # for d in dataSet[1:]:
    #     newDataSet = np.concatenate((newDataSet, d), axis=0)
        # newDataSet = np.vstack((newDataSet,d))
        # newDataSet = np.insert(newDataSet, 0, values=d, axis=0)

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
    shuffle_index = np.random.permutation(len(X))
    test_length = int(RATIO * len(shuffle_index))
    test_index = shuffle_index[:test_length]
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


# 构建CNN模型
def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300, 12)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        # tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='SAME', activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        # tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        # tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='SAME', activation='relu'),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.1),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return newModel


# 混淆矩阵
def plotHeatMap(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
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
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = loadData()

    # np.save("X_train.npy", X_train)
    # np.save("Y_train.npy", Y_train)
    # np.save("X_test.npy", X_test)
    # np.save("Y_test.npy", Y_test)
    #
    # X_train = np.load("X_train.npy")
    # Y_train= np.load("Y_train.npy")
    # X_test= np.load("X_test.npy")
    # Y_test= np.load("Y_test.npy")

    if os.path.exists(model_path):
    # if False:
        # 导入训练好的模型
        model = tf.keras.models.load_model(filepath=model_path)
    else:
        # 构建CNN模型
        model = buildModel()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # 定义TensorBoard对象
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # 训练与验证
        model.fit(X_train, Y_train, epochs=250,
                  batch_size=128,
                  validation_split=RATIO,
                  callbacks=[tensorboard_callback])
        model.save(filepath=model_path)

    # 预测
    Y_pred = model.predict_classes(X_test)
    # 绘制混淆矩阵
    plotHeatMap(Y_test, Y_pred)


if __name__ == '__main__':
    main()
