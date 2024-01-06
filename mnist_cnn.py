import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,  BatchNormalization,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
#模型架构
def create_cnn():
    model = Sequential()
    # 第一个卷积层，8个卷积核，每个核大小为(3, 3)，激活函数为ReLU
    model.add(Conv2D(8, (3, 3), padding="same", input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # 第二个卷积层，16个卷积核，每个核大小为(3, 3)，激活函数为ReLU
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # 第三个卷积层，32个卷积核，每个核大小为(4, 4)，激活函数为ReLU
    model.add(Conv2D(32, (4, 4), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # 第四个卷积层，64个卷积核，每个核大小为(4, 4)，激活函数为ReLU
    model.add(Conv2D(64, (4, 4), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    # 平铺层，将多维数据展平成一维
    model.add(Flatten())
    # 全连接层，1024个神经元，激活函数为ReLU
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # 输出层，10个神经元（对应10个类别），激活函数为Softmax
    model.add(Dense(10, activation='softmax'))
    # 编译模型，使用Adam优化器，交叉熵损失函数，监测准确率
    model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])
    return model
def plot_acc_loss(loss,acc,val_loss,val_acc):
    """
    绘制训练过程中损失值和准确率的变化趋势图。
    参数:
    - loss: 训练集的损失值列表
    - acc: 训练集的准确率列表
    - val_loss: 验证集的损失值列表
    - val_acc: 验证集的准确率列表
    """
    f,ax=plt.subplots(2,1,figsize=(10,10))
    ax[0].plot(loss,color='b',label='Training Loss')
    ax[0].plot(val_loss,color='r',label='Validation Loss')
    ax[0].legend()
    ax[1].plot(acc,color='b',label='Training accuracy')
    ax[1].plot(val_acc,color='r',label='Validation accuracy')
    ax[1].legend()
    plt.show()
    return
def plot_picture(x):
    """
    绘制给定数据的一维和二维表示。
    参数:
    - x: 输入数据
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].plot(x[0])
    ax[0].set_title('784x1 data')
    # 在第二个子图中绘制二维表示的数据，使用灰度图像显示
    ax[1].imshow(x[0].reshape(28, 28), cmap='gray')
    ax[1].set_title('28x28 data')
    # 显示图形
    plt.show()
    return
def plot_test_images(images, y_pred):
    """
    绘制测试集图像及其对应的预测标签。
    参数:
    - images: 包含图像数据的列表或数组
    - y_pred: 模型的预测标签列表
    注意：假设图像是灰度图。
    """
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i][:, :, 0], cmap='gray')  # Assuming images are grayscale
        plt.axis('off')
        plt.title(f"Predicted: {y_pred[i]}")
    plt.show()
    return

if __name__ == '__main__':
    # 文件路径设置
    train_file = "./digital_recognizer/train.csv"  # 训练数据文件路径
    test_file = "./digital_recognizer/test.csv"    # 测试数据文件路径
    output_file = "submission.csv"     # 输出文件路径

    # 从指定文件加载训练数据，skiprows=1 表示跳过第一行（通常为列名）
    # dtype='int' 指定数据类型为整数，delimiter=',' 表示数据之间以逗号分隔
    raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
    # 从文件中加载 MNIST 测试集数据
    mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
    # 使用 train_test_split 函数将数据集拆分为训练集和验证集
    # raw_data[:,1:] 表示特征数据，raw_data[:,0] 表示标签数据
    # test_size=0.1 表示将 10% 的数据分配给验证集，90% 用于训练集
    x_train, x_val, y_train, y_val = train_test_split(
        raw_data[:, 1:], raw_data[:, 0], test_size=0.1)
    #展示数据
    plot_picture(x_train[0])

    # 将数据的形状变形为 (-1, 28, 28, 1)，其中 -1 表示自动推断，1 表示通道数为1（灰度图像）
    # 然后再通过除以 255 进行归一化，将像素值缩放到 0 到 1 之间
    x_train = (x_train.reshape(-1, 28, 28, 1)).astype("float32") / 255.
    x_val = (x_val.reshape(-1, 28, 28, 1)).astype("float32") / 255.
    x_test = (mnist_testset.astype("float32").reshape(-1, 28, 28, 1) / 255)
    # 将标签转为独热编码形式
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # 创建模型
    model = create_cnn()
    # 创建一个图像数据生成器（ImageDataGenerator）对象，进行数据增强
    datagen = ImageDataGenerator(zoom_range=0.1,  # 随机缩放图像的范围，以便引入变焦效果
                                 height_shift_range=0.1,  # 随机垂直平移图像的范围，引入高度方向的平移
                                 width_shift_range=0.1,  # 随机水平平移图像的范围，引入宽度方向的平移
                                 rotation_range=10)  # 随机旋转图像的范围，以度为单位
    # 创建学习率调度器（LearningRateScheduler）对象
    # 使用指数衰减函数 lambda x: 1e-3 * 0.9 ** x 调整学习率
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    # 使用 fit 函数进行模型训练
    hist = model.fit(
        datagen.flow(x_train, y_train, batch_size=16),  # 使用数据生成器进行数据增强
        steps_per_epoch=500,  # 每个 epoch 的步数，根据实际情况调整
        epochs=20,  # 训练的总 epoch 数
        verbose=2,  # 控制输出信息的详细程度，2 表示每个 epoch 输出一行信息
        validation_data=(x_val[:400, :], y_val[:400, :]),  # 验证集数据，用于在训练过程中评估模型性能
        callbacks=[annealer]  # 使用学习率调度器作为回调函数
    )
    # 使用 evaluate 函数评估模型在验证集上的性能
    final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
    # 输出最终的损失值和准确率
    print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
    #展现训练过程正确率和损失率
    plot_acc_loss(hist.history['loss'], hist.history['acc'], hist.history['val_loss'], hist.history['val_acc'])
    # 使用模型对验证集进行预测
    y_hat = model.predict(x_val)
    # 从预测结果中取最大值所在的索引，得到预测类别
    y_pred = np.argmax(y_hat, axis=1)
    # 从验证集的真实标签中取最大值所在的索引，得到真实类别
    y_true = np.argmax(y_val, axis=1)
    # 使用混淆矩阵评估模型性能
    cm = confusion_matrix(y_true, y_pred)
    # 输出混淆矩阵
    print(cm)
    # 使用 classification_report 函数生成分类报告
    print(classification_report(y_true, y_pred))
    #绘制测试集图像及其对应的预测标签。
    plot_test_images(x_val, y_pred)

    # 进行测试集的预测
    y_hat = model.predict(x_test, batch_size=64)
    # 创建比赛需要的预测结果文件
    with open(output_file, 'w') as f:
        f.write('ImageId,Label\n')
        for i in range(len(y_pred)):
            f.write("".join([str(i + 1), ',', str(y_pred[i]), '\n']))
