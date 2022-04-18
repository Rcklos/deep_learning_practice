#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 
@Date        : 2021/12/08 16:32:20
@Author      : ifish
@version     : 1.0
'''
# 添加相对路径
import os
import sys
sys.path.append(os.path.abspath('.'))

from PIL import Image
from random import sample
import numpy as np


trains_data_base = './dataset/fer2013/fer2013_train/'
test_data_base = './dataset/fer2013/fer2013_test/'
valida_data_base = './dataset/fer2013/fer2013_val/'
label_mapper = ['生气', '厌恶', '惊恐', '开心', '伤心', '惊讶', '中性']

class DataSet:
    def __init__(self,
        train_path=trains_data_base, 
        test_path=test_data_base, 
        valida_path=valida_data_base,
        trains_num = 0, test_num = 0, label_converto_str=False, image_size = None) -> None:
        '''
        @description:
        ------------
        @param: 测试集和训练集中每一个标签中的图片数量
        ------
        @Returns:
        --------
        '''
        self.trains_img = []
        self.test_img  = []
        self.trains_label = []
        self.test_label = []
        self.val_imgs = []
        self.val_label = []
        self.trains_num = trains_num
        self.test_num = test_num
        self.label_converto_str = label_converto_str

        global trains_data_base
        global test_data_base
        global valida_data_base
        if train_path != trains_data_base:
            trains_data_base = train_path
        if test_path != test_data_base:
            test_data_base = test_path
        if valida_path != valida_data_base:
            valida_data_base = valida_path

    def get_data_set(self, is_separate=False, image_shape=None):
        '''
        @description: 现在还没有做训练集与测试集分离
        ------------
        @param: 训练集和测试集是否分开
        ------
        @Returns:
        --------
        '''
        # 读取trains
        labels = os.listdir(trains_data_base)
        self.trains_set = []
        for label in labels:
            label_path = trains_data_base + '/' + label
            images = os.listdir(label_path)
            sample_list = sample(images, self.trains_num if self.trains_num < len(images) else len(images))
            for image in sample_list:
                im = Image.open(label_path + '/' + image)
                if not image_shape is None:
                    im = im.resize(image_shape)
                self.trains_img.append(np.array(im))
                if self.label_converto_str:
                    self.trains_label.append(label_mapper[int(label)])
                else:
                    self.trains_label.append(int(label))
        #     print(sample_list)
        # print(self.trains_label)

        
        if is_separate:
            test_data_base_path = test_data_base
        else:
            test_data_base_path = trains_data_base
        # 读取test
        labels = os.listdir(test_data_base_path)
        for label in labels:
            label_path = test_data_base_path + '/' + label
            images = os.listdir(label_path)
            sample_list = sample(images, self.test_num if self.test_num < len(images) else len(images))
            for image in sample_list:
                im = Image.open(label_path + '/' + image)
                if not image_shape is None:
                    im = im.resize(image_shape)
                self.test_img.append(np.array(im))
                if self.label_converto_str:
                    self.test_label.append(label_mapper[int(label)])
                else:
                    self.test_label.append(int(label))
        #     print(sample_list)
        # print(self.test_label)
        return self.trains_img, self.trains_label, self.test_img, self.test_label

    def get_data_set_with_val(self, image_shape=None):
        self.get_data_set(image_shape=image_shape)
        if valida_data_base:
                # 读取val
                labels = os.listdir(valida_data_base)
                for label in labels:
                    label_path = valida_data_base + '/' + label
                    images = os.listdir(label_path)
                    sample_list = sample(images, len(images))
                    for image in sample_list:
                        im = Image.open(label_path + '/' + image)
                        if not image_shape is None:
                            im = im.resize(image_shape)
                        self.val_imgs.append(np.array(im))
                        if self.label_converto_str:
                            self.val_label.append(label_mapper[int(label)])
                        else:
                            self.val_label.append(int(label))
        
        return self.trains_img, self.trains_label, self.test_img, self.test_label, self.val_imgs, self.val_label
    

if __name__ == '__main__':
    dataset = DataSet(trains_num=5, test_num=5)
    set = dataset.get_data_set_with_val((64, 64))[0][0]
    print(type(set))
    print(set.shape)
