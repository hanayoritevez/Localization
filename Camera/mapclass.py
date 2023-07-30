import cameraclass as cs


class MapPPoint():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.kfID = 0
        self.kflayer = 0
        self.kpID = 0


class SLAMMap():
    def __init__(self):
        self.KeyFrames = []
        self.MapPoints = []
        
        
    def SetKeyFrame(self,kfp):
        self.KeyFrames.append(kfp)