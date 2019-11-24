#coding:utf-8
#import sys
#sys.path.append(".")
import collections
import matplotlib.pyplot as plt
import numpy as np
import smpl.smpl_np as mysmpl
import time
from copy import deepcopy
import math
from joints.Get_joint import *
from contours.Get_contours import *
from projection import *
from smpl.lbs import global_rigid_transformation

def complexjoint_term(dofs, dimension, in_vertex,weights,rawjoint,s):
    AtA = np.zeros((dimension, dimension))
    AtB = np.zeros((dimension,))
    for v in range(len(in_vertex)):
        #line = lines[v]
        w = weights[v]
        t = in_vertex[v]
        #    in_vertex[v] #t是1*4矩阵，3Djoint的齐次坐标
        t[2]=0
        #t = np.array([in_vertex[v,0],in_vertex[v,1],0,1])

        r = np.array([rawjoint[v][0],rawjoint[v][1],0]) #openpose 2djoint
        temp = np.array([
            [1, 0, 0, 0, t[2], -t[1]],
            [0, 1, 0, -t[2], 0, t[0]],
            [0, 0, 1, t[1], -t[0], 0]])

        A = np.zeros((3, dimension))

        # for d, d_t in zip(dofs,x):
        #     if len(d) != len(d_t):
        #         print("自由度数目错误！！！")
        flag = 0
        for b in dofs:
            for i in b[1]:
                #print dof_order[int(flag/3)-2][0][0]
                # out_v = np.add(out_v, np.dot(temp,dofs[b][i])*w[b]*x[b][i])
                #print w[dof_order[int(flag/3)][0][0]]
                A[:, flag][:3] = np.dot(temp, i)*w[dofs[int(flag/3)][0][0]]
                #if(v==12):
                  #print w[dof_order[int(flag/3)][0][0]]
                flag += 1

        #print A
        B = r - t
        AtA = AtA + np.dot(np.transpose(A), A) * s
        AtB = AtB + np.dot(np.transpose(A), B) * s

    #    AtB = np.dot(np.transpose(AtA),AtB)
    #    AtA = np.dot(np.transpose(AtA),AtA)

    return AtA, AtB




def simplesolveLasso(AtA, Atb, t):
    maxIter = 10000
    n = len(AtA)
    # // obtain the least square   求解Ax = b
    x = np.dot(np.linalg.pinv(AtA),Atb)

    # // Lasso with shooting algorithm
    # // Fu (1998). Penalized regression: the bridge versus the lasso
    for m in range(maxIter):
        x_old = x.copy()
        for i in range(n):
            s_0 = np.dot(AtA[i], x) - AtA[i, i] * x[i] + Atb[i]

            if s_0 > t:
                x[i] = (t - s_0) / (2 * AtA[i, i])
            else:
                if s_0 < -t:
                    x[i] = -(t + s_0) / (2 * AtA[i, i])
                else:
                    x[i] = 0

        if np.linalg.norm(x_old - x) < 1e-8:
            break

    return x

#####全局处理
S_jointTerm=100
d3_local_bones =[18,20] #[16,18,20,17,19,21,1,4,7,2,5,8,15] 目标点
#扩增doforder
mydof_order=[[[16],[0,1,2]],[[18],[0,1,2]]] #优化目标关节
dimension = 0
for d in mydof_order:
    for d1 in d[1]:
        dimension+=1


if __name__ == '__main__':
    mod = mysmpl.SMPLModel('../model/model.pkl')
    pose = np.zeros(mod.pose_shape)
    mod.set_params(beta=mod.beta, pose=pose, trans=mod.trans)
    mod.save_to_obj('../data/outputobj/test.obj')


    (_, A_global) = global_rigid_transformation(mod.pose, mod.J, mod.kintree_table,xp=np)
    #注意这个函数应该在tpose下调用应该才对
    local_joints, joint_weights = get_d3joint_and_weights_from_bone(mod.parent, d3_local_bones,pose,mod)

    local_bone_weights = load_skinning_weights(mod.parent, joint_weights)
    
    #weights代表了按照上面d3_local_bones的顺序所包含的节点权重
    weights=local_bone_weights
    vertex=local_joints
    vertex[0]=A_global[18][:, 3][:3]
    vertex[1]=A_global[20][:, 3][:3]

    tar=deepcopy(vertex)
    ########
    #调整目标点
    ########

    J18= A_global[18][:, 3][:3]
    J20= A_global[20][:, 3][:3]
    tar18=J18
    tar18[1]=tar18[1]+0.2
    tar20=[J18[0],J18[1]+(J20[0]-J18[0]),0]
    #tar20=[J18[0]+(J20[0]-J18[0])/1.414+10,J18[1]-(J20[0]-J18[0])/1.414-10,0]
    tar[0]=tar18
    tar[1] = tar20
    tar = points_resize(tar, 1000, [0, 100, 100])



    get_contour(mod.verts, mod.faces, 400)  # 可视化

    while (1):
        #mod.J = points_resize(mod.J, 1000, [0, 100, 100])

        (_, A_global) = global_rigid_transformation(mod.pose, mod.J, mod.kintree_table, xp=np)
        for i in range(len(vertex)):
            vertex[i] = A_global[d3_local_bones[i]][:, 3][:3]

        vertex = points_resize(vertex, 1000, [0, 100, 100])

        dofs = get_twist(A_global, mydof_order)


        AtA,Atb=complexjoint_term(dofs,dimension,vertex,weights,tar,S_jointTerm)
        #print AtA
        #print Atb
        #np.savetxt('AtA.txt',AtA)
        #np.savetxt('Atb.txt',Atb)
        #print np.linalg.lstsq(AtA, Atb)
        #np.linalg.inv(np.dot(np.transpose(AtA),AtA))*np.transpose(AtA)

        x=simplesolveLasso(AtA,Atb,1)


        #print np.dot(AtA,x)
        print x
        """
        pose[16][0]-=x[0]
        pose[16][1]-=x[1]
        pose[16][2]-=x[2]
        pose[18][0]-=x[3]
        pose[18][1]-=x[4]
        pose[18][2]-=x[5]
        """

        for i in range(dimension):

            pose[mydof_order[int(i/3)][0][0]][i%3] -= x[i]

        mod.set_params(beta=mod.beta, pose=pose, trans=mod.trans)

        mod.save_to_obj('../data/outputobj/test.obj')
        get_contour(mod.verts, mod.faces, 400)  # 可视化
#之前解出来的z正负的情况 对比 乘进A 看看结果
