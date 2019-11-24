#coding:utf-8
import collections
import matplotlib.pyplot as plt
import numpy as np
import cv2
import smpl_np as smpl
import time
import ctypes
from copy import deepcopy


def get_contour(vertices, faces, magnification):
    """
    mod.verts,mod.faces,放大倍率(recomand:500)
    """
    #print(img.shape[:2])

    im = np.zeros(((800,800)),np.uint8)
    # contour1 = np.zeros((img.shape[:2]),np.float32)
    #test = np.zeros((img.shape[:2]),np.uint8)
    #顶点在平面上的坐标
    coor = []
    #获取每个顶点的投影坐标

    vertices=vertices*magnification
    vertices[:,0]=vertices[:,0] +400
    vertices[:,1]=-vertices[:,1] +300

    for vertex in vertices:
        t = np.array([float(vertex[0]),float(vertex[1]),float(vertex[2]),1.])
        out_point = t


        coor.append([int(out_point[0]),int(out_point[1])])
    # for v in range(len(coor)-1):
    #     if abs(coor[v][0] - coor[v+1][0]) < 10 and abs(coor[v][1] - coor[v+1][1]) < 10:
    #         cv2.line(im,(coor[v][0],coor[v][1]),(coor[v+1][0],coor[v+1][1]),255,1)
    for f in faces:
        cv2.line(im,(coor[f[0]][0],coor[f[0]][1]),(coor[f[1]][0],coor[f[1]][1]),255,1)
        cv2.line(im,(coor[f[0]][0],coor[f[0]][1]),(coor[f[2]][0],coor[f[2]][1]),255,1)
        cv2.line(im,(coor[f[2]][0],coor[f[2]][1]),(coor[f[1]][0],coor[f[1]][1]),255,1)
    cv2.imshow('im',im)
    cv2.waitKey(0)
    return im



if __name__ == '__main__':
    #方便上传 把pkl文件放到上级目录下
    mod=smpl.SMPLModel('../model.pkl')
    im = get_contour(mod.verts,mod.faces,500)
    #cv2.imshow('im',im)
    #cv2.waitKey(0)

    #测试自写lib
    """
    x为一个list 表示点的横坐标
    X为转化为c_type的指针
    同理y
    返回的结果在X与Y中 
    读取方式与c++中读取数组相同
    存储着与(x[i],y[i])距离最近的目标点的坐标(X[i],Y[i])
    考虑指定优化的目标轮廓名字最好传递一个数字后缀名
    mask自动生成的前缀写死在库中 方便传参
    """
    ll = ctypes.cdll.LoadLibrary
    lib = ll("./get_dis_Point2contours.so")
    lib.test.restype = ctypes.c_float
    lib.fun.argtypes = [ctypes.POINTER(ctypes.c_int)]

    x = [1,2,3,4,5,6,7,8,9,10]

    X = (ctypes.c_int*len(x))(*x)
    
    y = [1,2,3,4,5,6,7,8,9,10]

    Y = (ctypes.c_int*len(y))(*y)

    qqq=lib.test(X,Y,len(x))


    for i in range(len(x)):
        print X[i]," ",Y[i]
    img=cv2.imread("002err1.jpg")
    cv2.imshow('img',img)
    cv2.waitKey(0)