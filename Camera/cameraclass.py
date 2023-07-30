import numpy as np
import cv2




class KeyFramePyramid():
    def __init__(self,layers):
        self.layerN = layers
        self.KFs = []
        for i in range(self.layerN):
            kf = KeyFrame()
            self.KFs.append(kf)
            self.pose = np.eye(4)
    def SetImage(self,Image):
        scale = 1.
        for i in range(self.layerN):
            self.KFs[i].SetImage(cv2.resize(Image,None, None,scale, scale, cv2.INTER_NEAREST))
            scale = scale*0.5
            
    def SetPose(self,Mtx):
        self.pose = Mtx
    
    def GetPose(self):
        return self.pose[3,0:3]
    
    def GetDirection(self):
        return self.pose[0:3,0:3]@np.reshape(np.array([0,0,1]),[3,1])
            
class KeyFrame():
    def __init__(self):
        # self.kp = np.zeros(max_nums,2,layers)
        # self.kp_num = np.zeros(layers,1)
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
                


    # for i in range(len(matchindex)):                
    #     print(matchvalue[i]) 
                    
    MatchSort(queindex,matchindex,matchvalue,0,len(matchindex))
    
    

    
                
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
        
        


def Get3DPoint(pts1,pts2_2img,M,K):
    #world 座標系を自己座標系に変換するための行列
    Mi = np.linalg.inv(M)
    pts1out = pts1

    pts1m = np.concatenate([np.array(pts1).reshape([-1,2,1]),np.ones((len(pts1),2, 1))],axis=1)
    
    #現在座標系に変換    
    pts1_2 = Mi@pts1m

    pts1_2 = pts1_2[:,0:3,:]
    
    olla = Mi[0:3,3]

    orignpt,jac = cv2.projectPoints(olla,np.array([0,0,0],"float64").reshape([3,-1]),np.array([0,0,0],"float64").reshape([3,-1]),K,None)
    ori =  np.array(orignpt).reshape([2,1])
    pts1_2img,jac = cv2.projectPoints(pts1_2.transpose(0,2,1),np.array([0,0,0],"float64").reshape([3,-1]),np.array([0,0,0],"float64").reshape([3,-1]),K,None)
    #　要はepipolar searchしている
    print(ori)
    print("ori")

    for i in range(pts1_2img.shape[0]):
        evec = np.array(pts1_2img[i]).reshape([2,1]) - ori
        
        print(evec)
        print(pts2_2img[i])
        # d = x - (t* evec + ori)  ddを  tで微分　d' = 0 は　 t = evec.(x-ori)/evec.evec
        q = evec.T@(np.array(pts2_2img[i]).reshape([2,1])- ori)/evec.T@evec
        pts1out[i] = q*pts1out[i]
        
        print(q)
        
    return pts1out
        
        