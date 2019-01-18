# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# ---------------------------------导包-------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sp


#---------------------------------导入文件---------------------------
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv') / 255
# sample_data = pd.read_csv('../data/sample_submission.csv')
# print(train_data.info())
# print(train_data.head())
# print(test_data.info())
print(test_data.head())
# print(sample_data.info())
# print(sample_data.head())
# 测试集输入
train_data_x = train_data.iloc[:, 1:] / 255
# train_data_x = np.multiply(train_data_x, 1.0 / 255)
# test_data = np.multiply(test_data, 1.0 / 255)
print(train_data_x.head())
# 测试集输出
encoder = sp.OneHotEncoder(sparse=False)
train_data_y = encoder.fit_transform(train_data.iloc[:, [0]])
# train_data_y = train_data.iloc[:,[0]]
print(type(train_data_y))
print(train_data_y.shape)
print(train_data_y[1:5])
# 图片尺寸
img_size = 28
# 扁平化特征
img_size_flat = train_data_x.shape[1]

# 图像维度
img_shape = [28, 28]
# 颜色通道
num_channels = 1
# 标签类别  #10
num_classes = 10


# --------------------------------卷积网络配置------------------------
def new_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.01))


def new_biases(num):
    return tf.Variable(tf.random_normal(shape=[num]))

# -------------------------------创建卷积层---------------------------


def new_conv_layer(input_data, input_channels, filter_size, filter_num, use_pool=True):
    '''
    args：
        前一层输入数据.
        前一层通道数
        卷积核尺寸
        卷积核数目
        使用 2x2 max-pooling.
    '''
    filter_shape = [filter_size, filter_size, input_channels, filter_num]
    # 生成卷积核
    weights = new_weights(filter_shape)
    # 卷积
    layer = tf.nn.conv2d(input_data, weights, strides=[
                         1, 1, 1, 1], padding='SAME')
    biases = new_biases(filter_num)
    layer += biases
    # 最大值池化
    print(layer.shape)
    if use_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[
                               1, 2, 2, 1], padding='SAME')
    # 激活函数,小于0的值变为0
    layer = tf.nn.relu(layer)
    print(layer.shape)
    return layer, weights

# --------------------------------展平---------------------------------
# 返回展平层、特征值维度


def flatten_layer(layer):
    shape = layer.get_shape()
    # 特征维度个数
    el_num = shape[1: 4].num_elements()
    layer = tf.reshape(layer, [-1, el_num])
    return layer, el_num

# ---------------------------创建全连接层------------------------------


def new_fc_layer(input_data, input_size, output_size, use_relu=True):
    fc_shape = [input_size, output_size]
    weights = new_weights(fc_shape)
    biases = new_biases(output_size)
    layer = tf.matmul(input_data, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

# -----------------------------参数---------------------------------
# 卷积核尺寸
filter_size = 5
# 卷积核1 的个数
filter1_num = 32
# 卷积核2 的个数
filter2_num = 64
# 全连接后特征数
fc_features = 1024

# -----------------------------占位符---------------------------------
x = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, shape=[-1, img_size, img_size, num_channels])
# y_true为各标签类别的概率
y_true = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='y')
# 概率最大的类别(下标)
y_true_cls = tf.argmax(y_true, axis=1)

# ----------------------------卷积----------------------------------
# 卷积层1
layer_conv1, weights_conv1 = new_conv_layer(
    x_image, num_channels, filter_size, filter1_num)
# 输入图像
# 输入通道数
# 卷积核尺寸
# 卷积核数目

# 卷积层2
layer_conv2, weights_conv2 = new_conv_layer(
    layer_conv1, filter1_num, filter_size, filter2_num)

# ---------------------------展平--------------------------------------
layer_flat, num_features = flatten_layer(layer_conv2)

# ---------------------------全连接------------------------------------
# 全连接层1  1024特征
layer_fc1 = new_fc_layer(layer_flat, num_features, fc_features)
# # 降低采样，防止过拟合
dropout = tf.layers.dropout(layer_fc1, rate=0.3)
# 全连接层2   10特征
layer_fc2 = new_fc_layer(layer_fc1, fc_features, num_classes, use_relu=False)

# ---------------------------预测类别----------------------------------
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# -------------------------代价函数，优化------------------------------
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=layer_fc2, labels=y_true)
lost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(lost)
# 性能度量
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ===================================执行================================
# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(15001):
    start = (i * 516) % len(train_data)
    end = start + 516
    # print(train_data_x.iloc[start: end,:])
    # print(train_data_x.iloc[start: end,:].shape)
    # print(train_data_y.iloc[start: end,:])
    # print(train_data_y.iloc[start: end,:].shape)
    # feed_dict_train = {x: train_data_x.iloc[start: end,:], y_true: train_data_y.iloc[start: end,:]}
    # print(train_data_y[start: end])
    # print(train_data_y[start: end].shape)
    feed_dict_train = {x: train_data_x.iloc[
        start: end, :], y_true: train_data_y[start: end]}
    sess.run(optimizer, feed_dict=feed_dict_train)
    if i % 100 == 0:
        print(i, '--->', sess.run(accuracy, feed_dict=feed_dict_train))
for i in range(7):
    print(i)
    pred_y = sess.run(y_pred_cls, feed_dict={
                      x: test_data.iloc[i * 4000:(i + 1) * 4000, :]})
    result = pd.DataFrame({'ImageId': np.arange(
        i * 4000 + 1, (i + 1) * 4000 + 1), 'Label': pred_y})
    # sess.run(result.to_csv('./predict_CNN.py',index=False),feed_dict={x:test_data})
    if i == 0:
        result.to_csv('./predict_CNN.csv', index=False)
    else:
        result.to_csv('./predict_CNN.csv', mode='a',
                      header=False, index=False)

sess.close()
