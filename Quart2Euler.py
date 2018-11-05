 ##四元数转Euler角##
import sys
import math
def quart2Euler (*nums):
   Euler = []
   w = float(nums[0])  #  img items: x,y,z   real items :w  #
   x = float(nums[1])
   y = float(nums[2])
   z = float(nums[3])
   theta1 = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
   if theta1 < 0:
     theta1 += 2 * math.pi
   temp = w * y - z * x
   if temp >= 0.5:
     temp = 0.5
   elif temp <= -0.5:
     temp = -0.5
   else:
     pass
   theta2 = math.asin(2 * temp)
   theta3 = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
   if theta3 < 0:
     theta3 += 2 * math.pi
   ##弧度转角度##
   roll = theta1 * 180 / math.pi
   Euler.append(roll)
   pitch = theta2 * 180 / math.pi
   Euler.append(pitch)
   yaw = theta3 * 180 / math.pi
   Euler.append(yaw)
   return Euler

if __name__ == '__main__':
  quart1 = (1,0,0,0)
  print(quart2Euler(*quart1))



