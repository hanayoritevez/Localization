import numpy as np
import cv2


class Pose():
    def __init__(self,x,y,z,rot_x,rot_y,rot_z):
        self.x = x
        self.y = y
        self.z = z
        self.nx = rot_x
        self.ny = rot_y
        self.nz = rot_z
        

class KeytFrame():
    def __init__(self,max_nums,layers,image_size):
        self.kp = np.zeros(max_nums,max_nums,layers)
        self.kp_num = np.zeros(layers,1)
        self.pos = Pose(0,0,0,0,0,0)
        self.image_size = image_size
        

class KeyFrames():
    def __init__(self,max_kf,max_nums,layers,image_size):
        self.keyframes = [[KeytFrame(max_nums,layers,image_size)]*max_kf]
        self.kf_num = 0