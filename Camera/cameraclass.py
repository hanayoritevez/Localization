import numpy as np
import cv2




class KeyFramePyramid():
    def __init__(self,layers):
        self.layerN = layers
        self.layeredImage = []
        for i in range(self.layerN):
            kf = KeyFrame()
            self.layeredImage.append(kf)
        self.pose = np.eye(4)
    def SetImage(self,Image):
        scale = 1.
        for i in range(self.layerN):
            akaze = cv2.AKAZE_create()
            self.layeredImage[i].SetImage(cv2.resize(Image,None, None,scale, scale, cv2.INTER_NEAREST))
            kp,des = akaze.detectAndCompute(self.layeredImage[i].image, None)
            self.layeredImage[i].SetKp(kp)
            self.layeredImage[i].SetKpDes(des)
    
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
        self.kpdes = None
       
        
    def SetImage(self,Image):
        self.image = Image
        self.image_size  = Image.shape
        
    def SetKp(self,kp):
        self.kp = kp
        
        
    def SetKpDes(self,kpdes):
        self.kpdes = kpdes
        
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
        if minindex >=0 and minval < 0.04:
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
        
        


            
        

def SE3_Jacobian(point,K):
    Jac = np.zeros([2,6])
    
    x = point[0,0]
    y = point[1,0]
    z = point[2,0]
    
    z2 = z*z
    
    Jac[0,0] = x*y/z2*K[0,0]
    Jac[0,1] = -(1+(x*x/z2))*K[0,0]
    Jac[0,2] = y/z*K[0,0]
    Jac[0,3] = -1./z*K[0,0]
    Jac[0,4] = 0.
    Jac[0,5] = x/z2*K[0,0]
    
    Jac[1,0] = (1+(y*y/z2))*K[1,1]
    Jac[1,1] = -x*y/z2*K[1,1]
    Jac[1,2] = -x/z*K[1,1]
    Jac[1,3] = 0.
    Jac[1,4] = -1./z*K[1,1]
    Jac[1,5] = y/z2*K[1,1]
    
    return Jac


def wv2Rt(wv):
    
    Wx = np.zeros([3,3])
    Wx[0,1] = -wv[2]
    Wx[0,2] =  wv[1]
    Wx[1,2] = -wv[0]
    Wx[1,0] =  wv[2]
    Wx[2,0] = -wv[1]
    Wx[2,1] =  wv[0]
    
    theta = np.linalg.norm(wv[0:3,0])
    
    R = np.eye(3)
    t = wv[3:6,0]
    if np.abs(theta) > 0.001: 
        R = np.eye(3) + np.sin(theta)/theta*Wx +(1. - np.cos(theta))/(theta*theta)* Wx@Wx
        V = np.eye(3) + (1. - np.cos(theta))/(theta*theta)*Wx +(theta - np.sin(theta))/(theta*theta*theta)* Wx@Wx
        t  = V@wv[3:6,0]
        
    return R,t


def GaussNewtonOptimizer(points,imgpoints,K):
    pointsit = points
    M = np.eye(4)
    R = np.eye(3)
    t = np.zeros([3,1])
    wv = np.array([0,0,0,0,0,0]).reshape([6,1])
    count = 0
    while True:
        dwv = GaussNewtonIteration(pointsit,imgpoints,K)
        # itration更新で姿勢更新
        wv = wv + dwv
        R,t = wv2Rt(wv)
        pointsit = R@points + t
        count += count
        if count > 20:
            break
        
    for ki in range(3):
            for kj in range(3):
                M[ki,kj]  = R[ki,kj]
                    
    for ki in range(3):
                M[ki,3] = t[ki]
                
    
    return  np.linalg.inv(M)

def GaussNewtonIteration(points,imgpoints,K):
    
    g = np.zeros([6,1])
    H = np.zeros([6,6])
    N = len(imgpoints)
    for i in range(N):
        point = points[i]
        imgpoint = imgpoints[i].reshape([2,1])
        
        prjpoint = K@(np.array([point[0,0]/point[0,2],point[0,0]/point[0,1],1.]).reshape([3,1]))
        e = prjpoint[0:2,0]-imgpoint
        J = SE3_Jacobian(point,K)
        H = H + (J.T@J)
        g = g + J.T@e
        
        dwv = -np.linalg.inv(H)@g
        
        
        return dwv
    

def EpipolarCheck(points,pts2_2img,M,K,distthr):
    ori,pts1_2img = GetEpipolarLine(points,M,K)
    mask = [0]*len(pts2_2img)
    for i in range(pts1_2img.shape[0]):
        evec = np.array(pts1_2img[i]).reshape([2,1]) - ori
        

        # d = x - (t* evec + ori)  ddを  tで微分　d' = 0 は　 t = evec.(x-ori)/evec.evec
        q = evec.T@(np.array(pts2_2img[i]).reshape([2,1])- ori)/(evec.T@evec)
        
        dist =  np.linalg.norm(q*evec + ori - pts2_2img[i])
        
        if distthr > dist:
            mask[i] = 1
            
    return mask

def GetEpipolarLine(points,M,K):
    Mi = np.linalg.inv(M)
    olla = Mi[0:3,3]
    points_2 = Mi@points
    orignpt,jac = cv2.projectPoints(olla,np.array([0,0,0],"float64").reshape([3,-1]),np.array([0,0,0],"float64").reshape([3,-1]),K,None)
    ori =  np.array(orignpt).reshape([2,1])
    pts1_2img,jac = cv2.projectPoints(points,np.array([0,0,0],"float64").reshape([3,-1]),np.array([0,0,0],"float64").reshape([3,-1]),K,None)
    
    
    return ori,pts1_2img
    
    


def Get3DPoint(pts1,pts2_2img,M,K):
    #world 座標系を自己座標系に変換するための行列
    Mi = np.linalg.inv(M)

    

    pts1m = np.concatenate([(np.array(pts1).reshape([-1,2,1])-K[0:2,2].reshape([2,1]))/K[0,0],np.ones((len(pts1),2, 1))],axis=1)
    pts1out = pts1m[:,0:3,:]
    #現在座標系に変換    
    pts1_2 = Mi@pts1m
    pts1_2 = pts1_2[:,0:3,:]
    
    olla = Mi[0:3,3]

    orignpt,jac = cv2.projectPoints(olla,np.array([0,0,0],"float64").reshape([3,-1]),np.array([0,0,0],"float64").reshape([3,-1]),K,None)
    ori =  np.array(orignpt).reshape([2,1])
    pts1_2img,jac = cv2.projectPoints(pts1_2.transpose(0,2,1),np.array([0,0,0],"float64").reshape([3,-1]),np.array([0,0,0],"float64").reshape([3,-1]),K,None)
    #　要はepipolar searchしている

    print("original point %f %f" % (ori[0],ori[1]))
    
    # print(pts1_2)
    # print(pts1_2img)
    for i in range(pts1_2img.shape[0]):
        evec = np.array(pts1_2img[i]).reshape([2,1]) - ori
        

        # d = x - (t* evec + ori)  ddを  tで微分　d' = 0 は　 t = evec.(x-ori)/evec.evec
        q = evec.T@(np.array(pts2_2img[i]).reshape([2,1])- ori)/(evec.T@evec)
        
        
        # カメラの並進ベクトルと　現在カメラの方向ベクトルの角度(sin)計算
        impt =  q*evec + ori
        vec2 = np.array([(impt[0,0]-K[0,2])/K[0,0],(impt[1,0]-K[1,2])/K[0,0],1]).reshape([3,1])
        # 基準カメラの方向ベクトル
        vec1 = pts1_2[i].reshape([3,1])-olla.reshape([3,1])

        # 正弦定理       
        cost = vec1.T@vec2/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        cos1 = olla.reshape([3,1]).T@vec2/(np.linalg.norm(olla)*np.linalg.norm(vec2))
        gain = np.linalg.norm(olla)*np.sqrt(1.-cos1*cos1)/np.sqrt(1.-cost*cost)
        print(pts1out[i] )

        pts1out[i] = gain*pts1out[i]/np.linalg.norm(pts1out[i])
        
        
        
        
    return pts1out,ori,pts1_2img
        
        