#coding:utf-8
import numpy as np
import pickle
from copy import deepcopy
from lbs import global_rigid_transformation
import cv2

def load_skinning_weights(parent, w_s):
    """
    加载顶点蒙皮权重到各个自由度
    输入：
        w_s: 不包含父级的蒙皮权重
    """
    # 父级组
    group = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 4, 5, 4, 5, 4, 5]
    out_w = []
    for w in w_s:
        bone_id = []
        bone_weight = []
        this_bone = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for b in w:
            if b[1] < 0.02:
                continue
            else:
                bone_id.append(int(b[0]))
                bone_weight.append(b[1])
        for id_, weight in zip(bone_id, bone_weight):
            i = id_
            this_bone[i] = this_bone[i] + weight
            if i != 0:
                while parent[i] != 0:
                    this_bone[parent[i]] = this_bone[parent[i]] + weight
                    i = parent[i]
            this_bone[0] = 1.0
        out_w.append(this_bone)
    return out_w


# 从旋转矩阵中获取旋转部分
def get_matrix_3_3(matrix):
  return matrix[0:3, 0:3]


# 从旋转矩阵中获取平移部分
def get_matrix_translation(matrix):
  return matrix[:, 3][:3]


def get_twist(bones, dof_order):
    """
    #根据每个骨骼矩阵求自由度的旋量表示
    输入：
        bones: 骨骼的矩阵信息
        dof_order: 每根骨骼取哪几个自由度
    输出：
        dofs: 每个自由度的旋量表示
    """
    dofs = []
    for i in dof_order:
        rotation = get_matrix_3_3(bones[i[0][0]])
        translation = get_matrix_translation(bones[i[0][0]])
        bone_dof = []
        for d in i[1]:
            v = np.cross(translation, rotation[:, d])
            dof = np.hstack((v, rotation[:, d]))
            bone_dof.append(dof)
        dofs.append([i[0], bone_dof])

    return np.array(dofs)

def get_d3joints(t_pose, trans, model):
    (_, A_global) = global_rigid_transformation(
        t_pose, model.J, model.kintree_table, xp=np)
    Jtr = np.vstack([g[:3, 3] for g in A_global])
    Jtr = Jtr + trans

    return Jtr

def get_d3joint_and_weights_from_bone(parent, bones, t_pose, model):
    cal = np.zeros((model.pose_shape))
    cal[1:] = t_pose[1:]
    (_, bone_mats) = global_rigid_transformation(
        cal, model.J, model.kintree_table, xp=np)
    joints = []
    weights = []
    bone_mat = np.zeros((4,4),np.float32)

    for bone in bones:
      if bone!= 12 and bone != 15:
        bone_mat = np.array(bone_mats[bone])
        joint = bone_mat[:, 3]
        joints.append(joint)
        weights.append([[parent[bone], 1.]])
    
    return joints, weights
