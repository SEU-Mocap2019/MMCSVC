import numpy as np
import cv2

a=np.loadtxt("002_keypoints.txt", dtype=float, delimiter=',')
print(a.shape)
b=[5,6,7,2,3,4,11,12,13,8,9, 10, 0, 1]
   # 0 1 2 3 4 5 6   7  8 9 10 11  12  13
c=[]
for i in b:
    c.append([a[i*3],a[i*3+1]])
print c
# np.savetxt('openposejoint.txt',c)

cv2.waitKey(0)