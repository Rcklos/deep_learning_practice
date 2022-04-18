#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 神经网络框架
@Date        : 2021/12/11 04:23:47
@Author      : ifish
@version     : 1.0
'''
import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./dataset'))
from collections import OrderedDict
from layer.layer import *
import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Activation, SeparableConv2D, Input, MaxPooling2D, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
print('tensorflow version: ', tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

class TowLayerNet:
    def __init__(self, input_size, hiden_size, output_size, weight_init_std=0.01) -> None:
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hiden_size) / np.sqrt(2 / hiden_size)
        self.params['b1'] = np.zeros(hiden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hiden_size, output_size) / np.sqrt(2 / hiden_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer(self.params['w2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def saveNetWork(self, path):
        pass
    
    def gradient(self, x, t):
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self,x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

class TensorFlowTowLayerNetWork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, 
                    model_path = './model/fer2013_model.h5') -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weight_init_std = weight_init_std
        self.model_path = model_path
        self.is_load = False
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print('load model')
            self.is_load = True
        else:
            self.model = self.create_model((48, 48))
            
        # self.model.build(input_shape=(None, input_size))
        self.model.summary()

    def create_model(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(self.input_size, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(self.output_size, activation='softmax')
        ])

        model.compile(optimizer='adagrad',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False),
            metrics=['categorical_accuracy'])

        return model

    def train(self, x_train, y_train, batch_size, epochs, path='./tensor/weight.ckpt'):
        index = [i for i in range(len(x_train))]
        np.random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]
        checkpoint_path = path
        if os.path.exists(checkpoint_path + '.index'):
            self.model.load_weights(checkpoint_path)
            print('load weights')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[cp_callback], validation_split=0.2)
        self.model.summary()
        self.model.save(self.model_path)

    def accuracy(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=2)

    def predict(self, input):
        return self.model.predict(input)
        
class TensorFlowXceptionNetwork:
    def __init__(self, input_shape, num_classes, l2_regularization=0.01):
        self.model = self.create_model(input_shape, num_classes, l2_regularization)
        self.model.compile(optimizer="adam", 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False),
            metrics=['categorical_accuracy'])
        self.model.summary()

    def train(self, x_train, y_train, batch_size, epochs, 
        x_val, y_val,
        path='./tensor/xception_', 
        generator=None, patience=50):
        csv_logger = tf.keras.callbacks.CSVLogger('./temp/xception_.log', append=False)
        early_stop = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.1,
                                                        patience=int(patience/4), verbose=1)
        check_point = tf.keras.callbacks.ModelCheckpoint(path + '.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                                        verbose=1,
                                                        save_best_only=True)
        callbacks = [check_point, csv_logger, early_stop, reduce_lr]
        return self.model.fit(generator, 
                                steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                                verbose=1, callbacks=callbacks,
                                validation_data=(x_val, y_val))

    def create_model(self, input_shape, num_classes, l2_regularization=0.01):
        regularization = l2(l2_regularization)

        # base
        img_input = Input(input_shape)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # module 1
        residual = Conv2D(16, (1, 1), strides=(2, 2),
                        padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 2
        residual = Conv2D(32, (1, 1), strides=(2, 2),
                        padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 3
        residual = Conv2D(64, (1, 1), strides=(2, 2),
                        padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 4
        residual = Conv2D(128, (1, 1), strides=(2, 2),
                        padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        x = Conv2D(num_classes, (3, 3),
                # kernel_regularizer=regularization,
                padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        output = Activation('softmax', name='predictions')(x)

        model = Model(img_input, output)
        return model

class AdaGrad:
    def __init__(self, lr = 0.01) -> None:
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
        

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(5, input_shape=(3,)),
#     tf.keras.layers.Softmax()])
# nn = TensorFlowTowLayerNetWork(input_size=48*48, hidden_size=128,  output_size=7)
# model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(48, 48)),
#         tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
#         tf.keras.layers.Dense(7, activation='softmax')
#     ])
# model.compile(optimizer='adagrad',
#             loss=tf.keras.losses.CategoricalCrossentropy(from_logits = False),
#             metrics=['categorical_accuracy'])
# model = TensorFlowTowLayerNetWork(input_size=48*48, hidden_size=128,  output_size=7).create_model((48, 48))
# model.save('./model/model2')
# loaded_model = tf.keras.models.load_model('./model/model2')
# x = tf.random.uniform((1, 48, 48))
# assert np.allclose(model.predict(x), loaded_model.predict(x))

# nn = TensorFlowXceptionNetwork(input_shape=(64, 64, 1), num_classes=7)
# print('end')