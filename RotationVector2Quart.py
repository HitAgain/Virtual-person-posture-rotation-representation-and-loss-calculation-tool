import math
import cv2
import numpy as np
def rotvector2quart(v):  ##旋转向量转四元数##
    v1 = np.array([[0], [0], [0]], dtype=float)
    v1[0][0] = v[0]
    v1[1][0] = v[1]
    v1[2][0] = v[2]
    R = cv2.Rodrigues(v1)[0]
    q = np.zeros(4)
    K = np.zeros([4, 4])
    K[0, 0] = 1 / 3 * (R[0, 0] - R[1, 1] - R[2, 2])
    K[0, 1] = 1 / 3 * (R[1, 0] + R[0, 1])
    K[0, 2] = 1 / 3 * (R[2, 0] + R[0, 2])
    K[0, 3] = 1 / 3 * (R[1, 2] - R[2, 1])
    K[1, 0] = 1 / 3 * (R[1, 0] + R[0, 1])
    K[1, 1] = 1 / 3 * (R[1, 1] - R[0, 0] - R[2, 2])
    K[1, 2] = 1 / 3 * (R[2, 1] + R[1, 2])
    K[1, 3] = 1 / 3 * (R[2, 0] - R[0, 2])
    K[2, 0] = 1 / 3 * (R[2, 0] + R[0, 2])
    K[2, 1] = 1 / 3 * (R[2, 1] + R[1, 2])
    K[2, 2] = 1 / 3 * (R[2, 2] - R[0, 0] - R[1, 1])
    K[2, 3] = 1 / 3 * (R[0, 1] - R[1, 0])
    K[3, 0] = 1 / 3 * (R[1, 2] - R[2, 1])
    K[3, 1] = 1 / 3 * (R[2, 0] - R[0, 2])
    K[3, 2] = 1 / 3 * (R[0, 1] - R[1, 0])
    K[3, 3] = 1 / 3 * (R[0, 0] + R[1, 1] + R[2, 2])
    D, V = np.linalg.eig(K)
    pp = 0
    for i in range(1, 4):
        if (D[i] > D[pp]):
            pp = i
    q = V[:, pp]
    q = np.array([q[3], q[0], q[1], q[2]])
    return q
def main():
    testvector = [1.57079633,0,0]
    print(rotvector2quart(testvector))
if __name__ == '__main__':
    main()

