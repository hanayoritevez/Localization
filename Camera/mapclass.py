import cameraclass as cs
import numpy as np


class MapPPoints():
    def __init__(self):
        self.X = []
        self.kfID = [] 
        self.kpID = [] # kframeの何個目のkpか
        self.kpdes = [] 
        
    def SetPoints(self,points):
        self.X.append(points)
        
    def SetKFIDs(self,kfid):
        self.kfID.append(kfid)
        
    def SetKPIDs(self,kpID):
        self.kpID.append(kpID)
        
    def SetKPDes(self,kpdes):
        self.kpdes.append(kpdes)
        
        
        
class MapPointsPyramid():
    def __init__(self,layers):
        self.layerN = layers
        self.layeredPoints = []
        for i in range(layers):
            mp = MapPPoints()
            self.layeredPoints.append(mp)
        
    def SetPoints(self,points,layer,kfid,kpid,kpdes):
        self.layeredPoints[layer].SetPoints(points)
        self.layeredPoints[layer].SetKPIDs(points)
        N = len(kpid)
        kfids = [kfid]*N
        self.layeredPoints[layer].SetKFIDs(kfids)
        
        self.layeredPoints[layer].SetKPDes(kpdes)
        

class SLAMMap():
    def __init__(self,layers):
        self.KeyFrames = []
        self.MapPoints = MapPointsPyramid(layers)
        self.layerN = layers
        
        
    def SetKeyFrame(self,img,pose):
        kfp = cs.KeyFramePyramid(self.layerN)
        kfp.SetImage(img)
        kfp.SetPose(pose)
        self.KeyFrames.append(kfp)
        
        
    def SetMapPoints(self,points,layer,kfid,kpid):
        kpdes = []
        N = len(kpid)
        for i in range(N):
           kpdes.append(self.KeyFrames[kfid].layeredImage[layer].kpdes[kpid])
        self.MapPoints.SetPoints(points,layer,kfid,kpid,kpdes)
        
    # 地図から現在のcameraに投影する →　各点に関してepipolar探索し　　マッチ度合いを見て対応点決定　　→対応点の誤差が最小になる用にRt計算
    # def ConstructPoints(self):
        