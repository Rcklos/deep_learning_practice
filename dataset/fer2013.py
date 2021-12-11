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


trans_data_base = './dataset/fer2013/'
test_data_base = './dataset/fer2013/'
label_mapper = ['生气', '厌恶', '惊恐', '开心', '伤心', '惊讶', '中性']

class DataSet:
    def __init__(self, trans_num = 0, test_num = 0, label_converto_str=False) -> None:
        '''
        @description:
        ------------
        @param: 测试集和训练集中每一个标签中的图片数量
        ------
        @Returns:
        --------
        '''
        self.trans_img = []
        self.test_img  = []
        self.trans_label = []
        self.test_label = []
        self.trans_num = trans_num
        self.test_num = test_num
        self.label_converto_str = label_converto_str

    def get_data_set(self, is_separate=False):
        '''
        @description: 现在还没有做训练集与测试集分离
        ------------
        @param: 训练集和测试集是否分开
        ------
        @Returns:
        --------
        '''
        # 读取trans
        labels = os.listdir(trans_data_base)
        self.trans_set = []
        for label in labels:
            label_path = trans_data_base + '/' + label
            images = os.listdir(label_path)
            sample_list = sample(images, self.trans_num if self.trans_num < len(images) else len(images))
            for image in sample_list:
                im = Image.open(label_path + '/' + image)
                self.trans_img.append(np.array(im))
                if self.label_converto_str:
                    self.trans_label.append(label_mapper[int(label)])
                else:
                    self.trans_label.append(int(label))
        #     print(sample_list)
        # print(self.trans_label)

        
        if is_separate:
            test_data_base_path = test_data_base
        else:
            test_data_base_path = trans_data_base
        # 读取test
        labels = os.listdir(test_data_base_path)
        for label in labels:
            label_path = test_data_base_path + '/' + label
            images = os.listdir(label_path)
            sample_list = sample(images, self.test_num if self.test_num < len(images) else len(images))
            for image in sample_list:
                im = Image.open(label_path + '/' + image)
                self.test_img.append(np.array(im))
                if self.label_converto_str:
                    self.test_label.append(label_mapper[int(label)])
                else:
                    self.test_label.append(int(label))
        #     print(sample_list)
        # print(self.test_label)
        return self.trans_img, self.trans_label, self.test_img, self.test_label
    

if __name__ == '__main__':
    dataset = DataSet(trans_num=5, test_num=5)
    print(dataset.get_data_set()[3])
