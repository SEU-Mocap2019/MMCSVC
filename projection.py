#coding:utf-8
import collections
import matplotlib.pyplot as plt
import numpy as np
import cv2
import smpl_np as smpl
import time
import ctypes
from copy import deepcopy

def points_resize(points, scale, offset):
    '''
    放缩节点 先放缩 再位移
    '''
    points = np.array(points)
    points = points * scale
    for i in range(points.shape[1]):
        points[:, i] = points[:, i] + offset[i]
    return points

#透视投影
def surface_projection(vertices, faces, joint, exter, intri, image, op):
    im = deepcopy(image)

    intri_ = np.insert(intri, 3, values=0., axis=1)
    temp_v = np.insert(vertices, 3, values=1., axis=1).transpose((1, 0))

    out_point = np.dot(exter, temp_v)
    print("out_point:", out_point)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = (out_point.astype(np.int32)).transpose(1, 0)
    max = dis.max()
    min = dis.min()
    t = 255./(max-min)

    img_faces = []
    i = 0
    j = 0
    for f in faces:
        color = int((dis[f[0]] - min)*t)
        point = out_point[f]
        im = cv2.polylines(im, [point], True, (0, 255-color, color), 2)

    temp_joint = np.insert(joint, 3, values=1., axis=1).transpose((1, 0))
    out_point = np.dot(exter, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1, 0)
    for i in range(len(out_point)):
        if i == op:
            im = cv2.circle(im, tuple(out_point[i]), 9, (0, 0, 255), -1)
        else:
            im = cv2.circle(im, tuple(out_point[i]), 9, (255, 0, 0), -1)

    cv2.namedWindow("mesh", 0)
    cv2.resizeWindow("mesh", int(512*1.5), int(384*1.5))
    cv2.moveWindow("mesh", 0, 0)
    cv2.imshow('mesh', im)
    cv2.waitKey()

    return out_point, im


if __name__ == '__main__':
    points_test = [[1.0, 2, 3], [1, 1, 1]]
    # points_test = np.array([[1.0, 2, 3],[1,1,1]])
    print (points_test)
    print (points_resize(points_test,0.5,[0,0,1]))
