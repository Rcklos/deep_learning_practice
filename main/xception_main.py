import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./dataset'))
import numpy as np
from matplotlib import lines
from matplotlib import pyplot as plt
from dataset.dataset import fetch_fer2013

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from network.network import TensorFlowXceptionNetwork

image_shape = (64, 64, 1)
# 图像增强
image_data_generator = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip=True
)
x_img = None
x_label = None
t_img= None
t_label = None
v_img = None
v_label = None

batch_size = 100
network = TensorFlowXceptionNetwork(input_shape=image_shape, num_classes=7)


def fetch_dataset():
    print('\r装配FER2013数据集...', end='')
    dataset = fetch_fer2013(x_num=999999, t_num=999999, image_shape=image_shape[:2])
    global x_img
    global x_label
    global t_img
    global t_label
    global v_img
    global v_label
    x_img = np.expand_dims(dataset['train_img'], -1)
    x_label = dataset['train_label']
    t_img = np.expand_dims(dataset['test_img'], -1)
    t_label = dataset['test_label']
    v_img = np.expand_dims(dataset['val_img'], -1)
    v_label = dataset['val_label']

    
    print('\r装配FER2013数据集完成!')


if __name__ == "__main__":
    fetch_dataset()
    history = network.train(x_img, x_label, batch_size, 
            len(x_img) // batch_size, 
            v_img, v_label,
            generator=image_data_generator.flow(x_img, x_label, batch_size=batch_size))
    
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./tensor/xception_train_acc.jpg')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./tensor/xception_train_loss.jpg')
    plt.show()

    