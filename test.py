from PIL import Image
import numpy as np
import tensorflow as tf
import os

data_dir = 'test'
model_path = 'model/image_model'

def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for name in os.listdir(data_dir):
        fpath = os.path.join(data_dir, name)
        fpaths.append(fpath)

        image = Image.open(fpath)
        print(image.size, image.format, image.mode)
        #data = np.asarray(image, dtype=np.float32)/255.0
        data = np.array(image)/255.0

        label = int(name.split('_')[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)
    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels

fpaths, datas, labels = read_data(data_dir)

# 计算有多少类图片
num_classes = len(set(labels))

#定义placeholder，存放输入和标签
datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])
#存放Dropout参数的容器， 训练时为0.25，测试时为0
dropout_placeholder = tf.placeholder(tf.float32)

# 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 定义卷积层, 60个卷积核, 卷积核大小为3，用Relu激活
conv2 = tf.layers.conv2d(pool1, 60, 3, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool2)

# 全连接层，转换为长度为100的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholder)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)

# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)
# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

# 用于保存和载入模型
#saver = tf.train.import_meta_graph('model/image_model.meta')
saver = tf.train.Saver()

with tf.Session() as sess:
    print('测试模式')
    saver.restore(sess, model_path)
    print('从{}载入模型 '.format(model_path))
    #label和名称的对照关系
    label_name_dict = {
        0:'巴士',
        1:'出租车',
        2:'货车',
        3:'家用轿车',
        4:'面包车',
        5:'吉普车',
        6:'运动型多功能车',
        7:'重型货车',
        8:'赛车',
        9:'消防车'
    }

    test_feed_dict = {
        datas_placeholder:datas,
        labels_placeholder:labels,
        dropout_placeholder:0
    }

    predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
    #真实label和模型预测label
    for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
        real_label_name = label_name_dict[real_label]
        predicted_label_name = label_name_dict[predicted_label]
        print('{}\t{} => {}'.format(fpath, real_label_name, predicted_label_name))

    correct_number = 0
    for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
        if real_label == predicted_label:
            correct_number += 1
    correct_rate = correct_number/200
    print('正确率：{:.2%}'.format(correct_rate))


