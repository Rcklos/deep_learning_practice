#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 
@Date        : 2021/12/17 03:40:43
@Author      : ifish
@version     : 1.0
'''
import os
import sys
sys.path.append(os.path.abspath('.'))

import cv2 as cv
import logging
import logging.config
import numpy as np
from network.network import TensorFlowTowLayerNetWork, TensorFlowXceptionNetwork


logging.config.fileConfig('./config/logger.config')
log = logging.getLogger('test')

# 摄像头
capture = cv.VideoCapture(0)
# 获取级联分类器
face_cascade = cv.CascadeClassifier(r'./model/haarcascade_frontalface_default.xml')
# 表情分类器
# network = TensorFlowTowLayerNetWork(input_size=48*48*1, hidden_size=128,  output_size=7)
network = TensorFlowXceptionNetwork.load_network(r'./tensor/xception_.65-1.01.hdf5')
# 表情
label_mapper = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def crop_and_resize(img, rect):
    child_img = img[ rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return child_img


def capture_faces(gray):
    return face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(3, 3))


def emotion_classify(face):
    img = np.asarray(face).reshape(1, 48, 48)
    eval = network.predict(img)
    emotion = label_mapper[np.argmax(eval)]
    return emotion


def loop():
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow('gray', gray)

    faces = capture_faces(gray)
    for face in faces:
        if face.any():
            # 处理
            # face[1] = face[1] - 30 if face[1] > 0 else 0
            # face[3] = face[3] + 60 if face[1] + face[3] < frame.shape[1] - 61 else frame.shape[1] - face[1]
            face_img = crop_and_resize(gray, face)
            if face_img.any():
                cv.imshow('face', face_img)
                resize_img = cv.resize(face_img, (48, 48))  
                cv.imshow('resize', resize_img)
                emotion = emotion_classify(resize_img)
                log.info(emotion)
                cv.putText(frame, emotion, (face[0], face[1]), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                cv.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 1)

    cv.imshow('frame', frame)


if __name__ == '__main__':
    while True:
        loop()
        if cv.waitKey(1) == ord('q'):
            break