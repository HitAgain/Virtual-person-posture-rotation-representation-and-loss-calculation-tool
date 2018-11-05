import math
import numpy as np
def Euler2quatern(*nums):  ##input : 欧拉角列表 [roll,pitch,yaw],返回一个四元数列表

    roll = nums[0] / 2
    pitch = nums[1] / 2
    yaw = nums[2] / 2

    w = math.cos(roll) * math.cos(pitch) * math.cos(yaw) + math.sin(roll) * math.sin(pitch) * math.sin(yaw)

    x = math.sin(roll) * math.cos(pitch) * math.cos(yaw) - math.cos(roll) * math.sin(pitch) * math.sin(yaw)

    y = math.cos(roll) * math.sin(pitch) * math.cos(yaw) + math.sin(roll) * math.cos(pitch) * math.sin(yaw)

    z = math.cos(roll) * math.cos(pitch) * math.sin(yaw) + math.sin(roll) * math.sin(pitch) * math.cos(yaw)
    qua = [w, x, y, z]
    return qua
def main():
    testeuler = [0, 0, 0]
    print(Euler2quatern(*testeuler))
if __name__ == '__main__':
    main()
