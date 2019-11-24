#coding:utf-8
import numpy as np
import cv2

def load_joints_json():
    """
    读取Openpose输出文件
    """
    load_data = np.loadtxt("../data/openpose/Openpose_keypoints.txt", dtype=float, delimiter=',')
    print(load_data.shape)
    custom_order = [5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 0, 1]
    # 0 1 2 3 4 5 6  7  8 9 10 11  12  13
    img_joints = []
    for i in custom_order:
        img_joints.append([load_data[i*3], load_data[i*3+1]])
    print img_joints
    # np.savetxt('openposejoint.txt',c)
    return img_joints

def load_mask_img():
    """
    读取Mask R CNN输出图片
    """
    return contours


if __name__ == '__main__':
    load_joints_json()
