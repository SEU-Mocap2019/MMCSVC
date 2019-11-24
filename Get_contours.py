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
    mod=smpl.SMPLModel('./model.pkl')
    im = get_contour(mod.verts,mod.faces,500)
    cv2.imshow('im',im)
    cv2.waitKey(0)