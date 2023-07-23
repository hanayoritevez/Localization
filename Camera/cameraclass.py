import numpy as np
import cv2


class Pose():
    def __init__(self,x,y,z,rot_x,rot_y,rot_z,theta):
        self.x = x
        self.y = y
        self.z = z
        self.nx = rot_x
        self.ny = rot_y
        self.nz = rot_z
    

class KeyFrame():
    def __init__(self):
        # self.kp = np.zeros(max_nums,2,layers)
        # self.kp_num = np.zeros(layers,1)
        self.pos = Pose(0,0,0,0,0,0,0)
        self.image_size = [0,0]
        self.image = None
        self.kp = None
        
    def SetImage(self,Image):
        self.image = Image
        self.image_size  = Image.shape
        
    def SetKp(self,kp):
        self.kp = kp
        
    def GetTemplate(self,i,size):
        mu = self.kp[i].pt[0]
        mv = self.kp[i].pt[1]
        sizea = size//2
        sizeb = size-sizea
        
        a1 = int(max([mu-sizea,0]))
        b1 = int(max([mv-sizea,0]))
        c1 = int(min([mv+sizeb,self.image_size[1]-1]))
        d1  = int(min([mu+sizeb,self.image_size[0]-1]))
        
        return self.image[a1:d1,b1:c1]
    


        
    # def CalcKPVector(self,i,j,Kinv):
    #     return  Kinv@(self.kp[i,:,j].reshape([2,1]))
    
    
    # def EpipolarMatch(point,vec,sp,K):
    # # 線を投影する
    #     pA = sp
    #     pB = sp + 100*vec
        
    #     qA = K@pA
    #     qB = K@pB
        
    #     point@(qA - qB)
        
        
# class KeyFrames():
#     def __init__(self,max_kf,max_nums,layers,image_size):
#         self.keyframes = [[KeytFrame(max_nums,layers,image_size)]*max_kf]
#         self.kf_num = 0
        



def AllMatching(image,kp,Keyframe,size,maxdistance):
    N = len(Keyframe.kp)
    matchindex = []
    matchvalue = []
    queindex = []
    Nk = len(kp)
    for i in range(N):
        tmpimg =  Keyframe.GetTemplate(i,size)
        
        minval = 10000000
        minindex = -1
        for j in range(Nk):
            mu = kp[j].pt[0]
            mv = kp[j].pt[1]
            sizea = size//2
            sizeb = size-sizea
            image_size = image.shape
            
            a1 = int(max([mu-sizea,0]))
            b1 = int(max([mv-sizea,0]))
            c1 = int(min([mv+sizeb,image_size[1]-1]))
            d1  = int(min([mu+sizeb,image_size[0]-1]))
            cntimg = image[a1:d1,b1:c1]

            if  cntimg.shape[0] ==size and cntimg.shape[1] ==size and tmpimg.shape[0] ==size and tmpimg.shape[1] ==size and maxdistance > np.hypot(float(Keyframe.kp[i].pt[0]-mu),float(Keyframe.kp[i].pt[1]- mv)): 
            
                res = cv2.matchTemplate(tmpimg,cntimg,cv2.TM_SQDIFF_NORMED)
                cnt_minval, cnt_maxval, cnt_minloc, cnt_maxloc = cv2.minMaxLoc(res)
                
                if cnt_minval < minval:
                    minval = cnt_minval
                    minindex = j
        if minindex >=0 and minval < 0.02:
            matchindex.append(minindex)
            matchvalue.append(minval)
            queindex.append(i)
                
    print(len(matchindex))

    # for i in range(len(matchindex)):                
    #     print(matchvalue[i]) 
                    
    MatchSort(queindex,matchindex,matchvalue,0,len(matchindex))
    
    
    print(matchvalue[0:10])
    
                
    return queindex,matchindex,matchvalue


def MatchSort(queindex,matchindex,matchvalue,startidx,endidx):
    
    if endidx-startidx >1:
        pivot = matchvalue[endidx-1]
        pivotidx = endidx-1
        
        j  = startidx
        for i in range(startidx,pivotidx):

            if pivot > matchvalue[i]:
                tmpvalue = matchvalue[i]
                matchvalue[i] = matchvalue[j]
                matchvalue[j] = tmpvalue
                    
                tmpidx = matchindex[i]
                matchindex[i] = matchindex[j]
                matchindex[j] = tmpidx
                    
                tmpidx = queindex[i]
                queindex[i] = queindex[j]
                queindex[j] = tmpidx
                    
                j = j+1
            
            
        tmpvalue = matchvalue[j]
        matchvalue[j] = matchvalue[pivotidx]
        matchvalue[pivotidx] = tmpvalue
                    
        tmpidx = matchindex[j]
        matchindex[j] = matchindex[pivotidx]
        matchindex[pivotidx] = tmpidx
                    
        tmpidx = queindex[j]
        queindex[j] = queindex[pivotidx]
        queindex[pivotidx] = tmpidx
   
        
        # print(pivotidx)

        MatchSort(queindex,matchindex,matchvalue,startidx,j)
        MatchSort(queindex,matchindex,matchvalue,j+1,endidx)
    
    