import sys
import math
import cv2
import numpy as np
"""
This is a quaternion class that calculates the quaternion to the rotation matrix, the rotation vector, 
and the rotation of the two quaternions based on the correlation method provided. 
This is a key step in the virtual human project.
© Weinan Gan. All Rights Reserved.
"""
class Quart():
    def __init__(self,*nums):
        self.w = float(nums[0])
        self.x = float(nums[1])
        self.y = float(nums[2])
        self.z = float(nums[3])
    def Quart2Euler(self):   #四元数计算欧拉角#
        Euler = []
        w = self.w
        x = self.x
        y = self.y
        z = self.z
        theta1 = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        if theta1 < 0:
            theta1 += 2*math.pi
        temp = w*y - z*x
        if temp >= 0.5:
            temp = 0.5
        elif temp <= -0.5:
            temp = -0.5
        else:
            pass
        theta2 = math.asin(2 * temp)
        theta3 = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
        if theta3 < 0:
            theta3 += 2*math.pi
        ##弧度转角度##
        roll = theta1 * 180 / math.pi
        Euler.append(roll)
        pitch = theta2 * 180 / math.pi
        Euler.append(pitch)
        yaw = theta3 * 180 / math.pi
        Euler.append(yaw)
        return Euler
    def Euler2quatern(*nums): ##input : 欧拉角列表 [roll,pitch,yaw]
        roll = nums[0] / 2
        pitch = nums[1] / 2
        yaw = nums[2] / 2
        w = math.cos(roll) * math.cos(pitch) * math.cos(yaw) + math.sin(roll) * math.sin(pitch) * math.sin(yaw)
        x = math.sin(roll) * math.cos(pitch) * math.cos(yaw) - math.cos(roll) * math.sin(pitch) * math.sin(yaw)
        y = math.cos(roll) * math.sin(pitch) * math.cos(yaw) + math.sin(roll) * math.cos(pitch) * math.sin(yaw)
        z = math.cos(roll) * math.cos(pitch) * math.sin(yaw) + math.sin(roll) * math.sin(pitch) * math.cos(yaw)
        qua = [w, x, y, z]
        return qua
    """
    The transform from quart to rotvector
    """

    def Quart2RotationMatrix(self):   ##四元数计算旋转矩阵##
        R = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        w = self.w
        x = self.x
        y = self.y
        z = self.z
        # calculate element of matrix
        R[0][0] = np.square(w) + np.square(x) - np.square(y) - np.square(z)
        R[0][1] = 2 * (x * y + w * z)
        R[0][2] = 2 * (x * z - w * y)
        R[1][0] = 2 * (x * y - w * z)
        R[1][1] = np.square(w) - np.square(x) + np.square(y) - np.square(z)
        R[1][2] = 2 * (w * x + y * z)
        R[2][0] = 2 * (x * z + w * y)
        R[2][1] = 2 * (y * z - w * x)
        R[2][2] = np.square(w) - np.square(x) - np.square(y) + np.square(z)
        return R
    def rotMat2rotvector(R):  ##旋转矩阵转旋转向量##
        vector = cv2.Rodrigues(R)[0]
        v =[]
        for i in range(3):
            v.append(vector[i][0])
        return v
    def Quart2RotationVector(self):   ##四元数计算旋转向量##
        R = self.Quart2RotationMatrix()
        v = Quart.rotMat2rotvector(R)
        return v

    """
    the transform from rotvector to quartern
    """

    def rotvector2rotMat(v):    ##旋转向量转旋转矩阵##
        v1=np.array([[0],[0],[0]],dtype=float)
        v1[0][0] = v[0]
        v1[1][0] = v[1]
        v1[2][0] = v[2]
        Matrix = cv2.Rodrigues(v1)
        return Matrix[0]
    def rotMat2quatern(R):     ##旋转矩阵转四元数###
        # this function can transform the rotation matrix into quatern
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
        # print(R)
        # print("***********")
        # print(K)
        D, V = np.linalg.eig(K)
        # print(K)
        pp = 0
        for i in range(1, 4):
            if(D[i] > D[pp]):
                pp = i
        # print(D[pp])
        # print(D)
        q = V[:, pp]
        q = np.array([q[3], q[0], q[1], q[2]])
        return q
    def rotvector2quart(v):        ##旋转向量转四元数##
        rotmatrix = Quart.rotvector2rotMat(v)
        q = Quart.rotMat2quatern(rotmatrix)
        return q


    """
    loss about VirtualHuman's action 
    """
    def calculateloss1(self,*nums):      ##第一种方法，input：一个四元数元组
        R1 = np.matrix(self.Quart2RotationMatrix())
        quart2 = Quart(*nums)
        R2 = np.matrix(quart2.Quart2RotationMatrix())
        reverse_R1 = R1.I
        R3 = reverse_R1 * R2     ##这里本应该用转置的，但是由于浮点数计算误差带来问题，故选择求逆函数##
        loss_vector = Quart.rotMat2rotvector(np.array(R3))
        loss1 =  (loss_vector[0][0]**2 + loss_vector[1][0]**2 + loss_vector[2][0]**2)**0.5
        return loss1
    def calculateloss2(self ,predic_v ):  ##第二种方法,input predic_v = [a,b,c]##
        truth_v = self.Quart2RotationVector()
        loss_v_list = []
        for i in range(3):
            loss_v_list.append(truth_v[i][0]-predic_v[i])
        loss2 = (loss_v_list[0]**2 + loss_v_list[1]**2 + loss_v_list[2]**2)**0.5
        return loss2

    def Dicar_trans(self,*nums1):   ###笛卡尔坐标转换 这里用不到  input:调用对象 旧坐标  output：新的坐标##
        Matrix = self.Quart2RotationMatrix()
        newcordinate = []
        oldcordinate = np.array(nums1)
        for i in range(len(Matrix)):
            newcordinate.append(np.dot(Matrix[i],oldcordinate))
        return newcordinate
def main():
    testquart = (-0.707100,0.707100,0.000000,0.000000)
    old = (0,0,1)
    quart = Quart(*testquart)
    print(quart.Quart2RotationVector())
    print(quart.Quart2RotationMatrix())
    print(quart.Dicar_trans(*old))
   

    """
    testquart = (-0.707100, 0.707100, 0.000000, 0.000000)
    testeuler = ((270.00109895412885/180)*math.pi, -0.0, 0.0)
    quart = Quart(*testquart)
    print("欧拉角为：",quart.Quart2Euler())
    print("四元数为：",Quart.Euler2quatern(*testeuler))
    """
    """
    predict_quart = (-0.707100,0.707100,0.000000,0.000000)
    truth_quart = (-0.707100,0.707100,0.000000,0.000000)
    quart = Quart(*predict_quart)
    print("lossFuncition1的损失为：",quart.calculateloss1(*truth_quart))
    """
    """
    testquart = (1, 0, 0, 0)
    quart = Quart(*testquart)
    print(quart.Quart2RotationVector())
    print(Quart.rotvector2quart([0,0,0]))
    v_test = [1,0,0]
    print("lossFunction2的损失为：",quart.calculateloss2(v_test))
    """

if __name__ == "__main__":
    main()










