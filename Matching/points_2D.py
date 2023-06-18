
import numpy as np
from ctypes.wintypes import HBITMAP
from curses.ascii import STX
from imp import C_EXTENSION
from locale import DAY_2, DAY_3
from re import A
from signal import valid_signals
from tkinter import E
from unittest.main import MAIN_EXAMPLES
from zlib import Z_BLOCK

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import bisect



class KF:

    def __init__(self,x,y):
        self.x = x
        self.y = y






class PF:
    def __init__(self,x,y):
        self.x = x
        self.y = y


class KDnode:

        def __init__(self,point,key, parent=0,axis=0,left_key=0,right_key=0):

            self.parent = parent
            self.left_child = left_key
            self.right_child = right_key
            self.data =  point
            self.key = key
            self.axis =  axis
            self.bottom =  0 

        
class KDtree:

    def __init__(self,Points):

        self.dim = Points.shape[1]
        self.Nodes = []

        self.top = self.generate(Points)

    def Get(self,key):

        return self.Nodes[key]


    def NearestNeighbor(self,point):
        nn_key,nn_radius = self.calcurate_radius(self.top,point)
        return self.Nodes[nn_key].data



    def NaiveNearestNeighbor(self,Points,point):
        dist = 100000
        ans2_point = Points[0]
        for i in range(Points.shape[0]):
            current_dist = np.linalg.norm( point -Points[i])
            if dist > current_dist:
                dist = current_dist
                ans2_point = Points[i]
        return ans2_point

    def generate(self,Points,parent=-1,depth=0):
        if len(Points) == 0:
            return -1
        key = len(self.Nodes)
        axis = depth%self.dim
        Points = Points[np.argsort(Points[:,axis])]
        # if only 2 points are remained, the right is empty 
        median = len(Points)//2
        
        self.Nodes.append(KDnode(Points[median],key,parent,axis))

        self.Nodes[key].left_child = self.generate(Points[0:median],key,depth+1)
        self.Nodes[key].right_child  =self.generate(Points[median+1:],key,depth+1)

        self.Nodes[key].bottom = (self.Nodes[key].left_child == -1) +  (self.Nodes[key].right_child== -1)
        
        
        return key



    def calcurate_radius(self,key,point):


        node = self.Nodes[key]

        nn_radius = 0
        nn_key = 0

        if node.bottom == 2:
            return key, np.linalg.norm(point - node.data)

        elif node.bottom == 1:
            nn_key, nn_radius = self.calcurate_radius(node.left_child,point)
            this_radius = np.linalg.norm(point- node.data)

            if this_radius >= nn_radius:
                return nn_key, nn_radius

            else:
                return key,this_radius

        distance_along_axis,left_judgement = self.judge_right(node,point)
        
        if left_judgement==1:
            nn_key, nn_radius = self.calcurate_radius(node.right_child,point)
        else:
            nn_key, nn_radius = self.calcurate_radius(node.left_child,point)

        if nn_radius >= np.abs(distance_along_axis):
            this_radius = np.linalg.norm(point- node.data)
            #Compare with itself
            if this_radius <= nn_radius:
                nn_radius = this_radius
                nn_key = key
            #Compre with opposit side
            oppose_key = 0
            oppose_radius = 0
            if left_judgement:
                oppose_key, oppose_radius = self.calcurate_radius(node.right_child,point)
            else:
                oppose_key, oppose_radius = self.calcurate_radius(node.left_child,point)

            if  oppose_radius <= nn_radius:
                nn_radius = oppose_radius
                nn_key = oppose_key

        return  nn_key,nn_radius


    #rule: this function only can be used when the node is not  bottom == 2 nor bottom == 1  
    def judge_right(self,Node,point):
        
        distance_along_axis =  point[Node.axis] - Node.data[Node.axis]
        if distance_along_axis >= 0:
            return distance_along_axis,1
        else:
            return distance_along_axis, 0



class Point_Map:

    def __init__(self,A):
        self.mPoints = A
        self.mytree = KDtree(A)

    def Matching(self,vPoints,perr=200000):

        count = 0
        err = np.linalg.norm(vPoints - np.apply_along_axis((lambda x:self.mytree.NearestNeighbor(x)),1,vPoints),2)


        Rt = np.eye(4)
        while(np.abs(perr-err)>0.00000000001 and count<100):
            print(err)
            perr = err
            mPoints= np.apply_along_axis((lambda x:self.mytree.NearestNeighbor(x)),1,vPoints)
            
            # fig = plt.figure()
            # ax = Axes3D(fig,auto_add_to_figure=False)
            # fig.add_axes(ax)
            # ax.plot(vPoints[:,0],vPoints[:,1],vPoints[:,2], marker=".",linestyle='None',color='b')
            # ax.plot(mPoints[:,0],mPoints[:,1],mPoints[:,2], marker=".",linestyle='None',color='r')
            # plt.show()

            R,t = self.svd_matching(vPoints,mPoints)
            vPoints = (R@vPoints.T + t).T
            err = np.linalg.norm(mPoints - vPoints,2)


            count = count + 1



            Rt_tmp = np.concatenate([np.concatenate([R,t],axis=1),np.array([[0,0,0,1]])],axis=0)


            Rt = Rt_tmp@Rt
            

            

        return Rt[0:3,0:3].reshape(3,3),Rt[3,0:3].reshape(3,1)
    
    def svd_matching(self,vPoints,mPoints):

        vg = np.average(vPoints,axis=0)
        mg = np.average(mPoints,axis=0)

        
        nvPoints = vPoints -vg
        nmPoints = mPoints -mg

        N = nvPoints.T@nmPoints
        U,s,Vt = np.linalg.svd(N,full_matrices=True)

        h = np.ones(N.shape[0])
        h[N.shape[0]-1] = np.linalg.det(Vt.T@U.T)
        R = Vt.T@np.diag(h)@U.T

        t = mg.reshape(3,-1) -R@vg.reshape(3,-1)



        return R,t





class Point_Voxel:


    def __init__(self,Points=None):
        
        
        self.Points = np.empty([0,2])
        self.mu =None
        self.Sigma_inv = None
        self.empty = 1
        self.dimension = np.array([1.5 ,1.5 ])
        self.Sigma = None

        if Points:
            
            self.Points = Points
            self.mu = np.average(self.Points,axis=0)
         
            if self.Points.shape[0] < 5:
                self.Sigma_inv = np.linalg.inv(np.diag((self.dimension/2.)*(self.dimension/2.)))
            else:
                self.Sigma_inv = np.linalg.inv(np.cov(self.Points.T))
                self.empty =0
                self.Sigma = np.cov(self.Points.T)

    def add_point(self,point):
        
        self.Points = np.concatenate([self.Points,np.reshape(point,[1,2])],0)
        self.mu = np.average(self.Points,axis = 0)
        
        if self.Points.shape[0] < 5:
            self.Sigma_inv = np.linalg.inv(np.diag((self.dimension/2.)*(self.dimension/2.)))
        else:
            # print(self.Points)
            # self.Sigma_inv = np.linalg.inv(np.cov(self.Points.T ) + 0.0001*np.eye(3))
            # print(self.Sigma_inv)

            self.Sigma_inv = np.linalg.inv(np.cov(self.Points.T ) )
            self.Sigma = np.cov(self.Points.T)


            
            
            self.empty =0



class Point_VoxelMap:
    def __init__(self,mPoints,voxel_size = [5,5]):
        
        max_x = [-100000,-100000]
        min_x = [100000,100000]
        for i in range(mPoints.shape[1]):
            X = mPoints[:,i]
            min_x[i] = np.min(X)
            max_x[i] = np.max(X)

        self.Voxels_step = np.array(voxel_size)
        self.Voxels_start = np.array(min_x)  - np.array(voxel_size)/2.
        self.Voxels_size =  np.apply_along_axis(lambda x,step: x//step + 1., 0 , max_x - self.Voxels_start,self.Voxels_step).astype(np.int32)
        self.Voxels_end = np.apply_along_axis(lambda x,y,step: y*step + x, 0 , self.Voxels_start,self.Voxels_size,self.Voxels_step)
        print("size")
        print(self.Voxels_size)

        self.Voxels_array =[]
        for i in range(self.Voxels_size[0]):
            ys = []
            for j in range(self.Voxels_size[1]):
                ys.append(Point_Voxel())            
            self.Voxels_array.append(ys)

        voxel_indexs =np.apply_along_axis(self.search_voxel,1,mPoints)
        for index,X in zip(voxel_indexs,mPoints):
            
            self.Voxels(index).add_point(X)


        self.d1 = 1.
        self.d2 = 1.


    def Matching(self,vPoints):
        P = np.array([[0.],[0.],[0.]])
        last_P = np.array([[0.],[0.],[0.]])
        pre_score = -100.
        score = -100.

        counter = 0
        # while(pre_score < score):
        while(counter<10):
            pre_score = score
            trans_vPoints = (self.step_R(P)@vPoints.T+ np.reshape(P[0:2,0],[2,1])).T
            voxel_indexs =np.apply_along_axis(self.search_voxel,1,trans_vPoints).tolist()
          
            P,score = self.ndt_matching(trans_vPoints,voxel_indexs,P)

            counter = counter + 1


            # if pre_score  > score:
            #     break

        return self.step_R(P),np.reshape(P[0:2,0],[2,1])


    def Voxels(self,index):

        return self.Voxels_array[index[0]][index[1]]

            
    
    def search_voxel(self,X):
        dindexes = []
        for i in range(2):
            Ds = X[i] -self.Voxels_start[i]
            De = self.Voxels_end[i] - X[i]
            if De < 0 or Ds < 0:
                dindexes.append(-1)
            else:
                dindex = Ds // self.Voxels_step[i]
                dindexes.append(int(dindex))
        return dindexes

            


    def ndt_matching(self,vPoints,voxel_indexs,P):
        # Xm =  self.step_R()@X.T+ self.P[3:6]

        
        # print(voxel_indexs.shape)

        # scores = np.apply_along_axis(self.calc_score,1,vPoints,voxel_indexs)

        # scores = np.array

        # Hs,gs = np.apply_along_axis(self.calc_Hg,1,vPoints,voxel_indexs,P)

        score = 0

        Hs = np.zeros([3,3])
        gs = np.zeros([3,1])
        for Xk,voxel_index in zip(vPoints,voxel_indexs):
            score = score + self.calc_score(Xk,voxel_index)
            H,g = self.calc_Hg(Xk,voxel_index,P)
            
            Hs = Hs + H
            gs = gs + g


        print("delta H")
        print(Hs)
        print(gs)


        if np.linalg.det(Hs)==0:
             return P,score
            

        delta_P = -np.linalg.inv(Hs)@gs


        print("score")
        print(score)
        print("delta P")
        print(delta_P)

        
        P = P + delta_P

        print("P")
        print(P)

        return P,score



    def  calc_score(self,X_k,voxel_index):



        if voxel_index[0] <0    or  voxel_index[1] <0 :
                
                return -0


        if self.Voxels(voxel_index).empty == 1:
            return -0


        
      
        
             
        score = -self.d1*np.exp(-self.d2/2. *(X_k -self.Voxels(voxel_index).mu).T@self.Voxels(voxel_index).Sigma_inv@(X_k -self.Voxels(voxel_index).mu))

        return score
        

    def step_R(self,P):


        p_rz = P[2,0]
        
       
        cz = np.cos(p_rz)
        sz = np.sin(p_rz)


        R = np.zeros([2,2])

        R[0,0] =  cz
        R[0,1] = -sz
        R[1,0] =  sz
        R[1,1] =  cz
      
        return R


    def  calc_Hg(self,X_k_raw,voxel_index,P):

        
        Hk = np.zeros([3,3])
        gk = np.zeros([3,1])


        if voxel_index[0] < 0 or voxel_index[1] < 0 :
        # if voxel_index[0] != 0 or voxel_index[1] != 0 :
            return Hk,gk
        if self.Voxels(voxel_index).empty == 1:

            return Hk,gk


        

        X_k = np.reshape(X_k_raw -self.Voxels(voxel_index).mu,[2,1])

       
       

        Dt = X_k.T@self.Voxels(voxel_index).Sigma_inv
        Maht = X_k.T@self.Voxels(voxel_index).Sigma_inv@X_k

        if Maht > 1.:
            Dt = X_k.T@self.Voxels(voxel_index).Sigma_inv/np.sqrt(Maht)*0.9
            X_k = X_k/np.sqrt(Maht)*0.9


        if  self.d2*np.exp(-self.d2/2.*Dt@X_k) >1 or self.d2*np.exp(-self.d2/2.*Dt@X_k) <0:


            return Hk,gk
        
        # print("raw:" + str(X_k_raw) +"mu:"+ str(self.Voxels(voxel_index).mu))
        Sigma_k_inv = self.Voxels(voxel_index).Sigma_inv
        Ht = self.calc_hessian(np.reshape(X_k_raw,[2,1]),P)
        Jt = self.calc_jacobian(np.reshape(X_k_raw,[2,1]),P)


        
        gk  =  (np.exp(-self.d2/2.*Dt@X_k)*self.d1*self.d2*Dt@Jt).T
        
        # print(-self.d2/2.*Dt@X_k)
        # print(np.exp(-self.d2/2.*Dt@X_k))
        # print(self.d1*self.d2*Dt@Jt)
        # print("grad:"+str(gk))

        for i in range(3):
            for j in range(3):
                Hk[i,j] = self.d1*self.d2*np.exp(-self.d2/2.*Dt@X_k)*(-self.d2* (Dt@Jt[:,i])*(Dt@Jt[:,j]) + Dt@Ht[:,i,j] + Jt[:,j].T@Sigma_k_inv@Jt[:,i])

                print((Dt@Jt[:,i]))
                print(-self.d2* (Dt@Jt[:,i])*(Dt@Jt[:,j]))
                print(Sigma_k_inv@Jt[:,i])
        return Hk,gk
        


    def calc_jacobian(self, X,P):
        
        x = X[0,0]
        y = X[1,0]
        p_rz = P[2,0]
        
     
        cz = np.cos(p_rz)
        sz = np.sin(p_rz)

        Ja = -x*sz -y*cz
        Jb = x*cz -y*sz
      
        J = np.zeros([2,3])
        J[0,0] = 1.
        J[1,1] = 1.
        J[0,2] = Ja #z
        J[1,2] = Jb #-y


        
  
        return J 

    def calc_hessian(self, X,P):

        x = X[0,0]
        y = X[1,0]
        
        p_rz = P[2,0]

        cz = np.cos(p_rz)
        sz = np.sin(p_rz)

        H = np.zeros([2,3,3])

        H[0,2,2] = -x*cz + y*sz
        H[1,2,2] = -x*sz -y*cz

        return H

def Points_Generator():


    Points_xz1 = 10.*np.random.rand(1000,2)
    Points_xz1[:,1] = 0.01*Points_xz1[:,1]
    Points_xz2 = 10*np.random.rand(1000,2)
    Points_xz2[:,1] = 10.+ 0.01*Points_xz2[:,1]
    Points_yz = 10*np.random.rand(1000,2)
    Points_yz[:,0] = 10.+ 0.01*Points_yz[:,0]


    Points = np.concatenate([Points_xz1,Points_xz2,Points_yz])

    vPoints_xz1 = 10.*np.random.rand(200,2)
    vPoints_xz1[:,1] = 0.01*vPoints_xz1[:,1]
    vPoints_xz2 = 10*np.random.rand(200,2)
    vPoints_xz2[:,1] = 10.+ 0.01*vPoints_xz2[:,1]
    vPoints_yz = 10*np.random.rand(200,2)
    vPoints_yz[:,0] = 10.+ 0.01*vPoints_yz[:,0]
    vPoints = np.concatenate([vPoints_xz1,vPoints_xz2,vPoints_yz])
    for i in range(vPoints.shape[0]):
        vPoints[i] = vPoints[i] + np.random.randn(1,2)*0.1
    print(vPoints)

    return Points,vPoints 




def Rt_Generator(bx = 1,by=1,bz = 0):



    T = np.random.rand(2,1)*0.5

    T[0] = T[0]*bx
    T[1] = T[1]*by

    r = np.arcsin(np.random.rand(1)*0.3)


    Rz = np.zeros([2,2])

    Rz[0,0] = np.cos(r)
    Rz[1,1] = np.cos(r)
    Rz[0,1] = -np.sin(r)
    Rz[1,0] = np.sin(r)

    print(T)
    print(r)


    return Rz,T




# Points = np.random.rand(100000,3)


# Calcuration speed test
# start = time.time()
# mytree = KDtree(Points)
# end = time.time()
# print('KDtree generation time:' + str(end-start))
# start = time.time()
# for i in range(e_Points.shape[0]):
#     ans_point = mytree.NearestNeighbor(e_Points[i])
# end = time.time()
# print('KDtree calcuration time:' + str(end-start))
# start = time.time()
# for i in range(e_Points.shape[0]):
#     mytree.NaiveNearestNeighbor(Points,e_Points[i])
# end = time.time()
# print('Simple calcuration time:' + str(end-start))
#Visualize
# fig = plt.figure()
# ax = Axes3D(fig,auto_add_to_figure=False)
# fig.add_axes(ax)
# ax.plot(Points[:,0],Points[:,1],Points[:,2],marker=".",linestyle='None',color='b')

# ax.plot(point[0],point[1],point[2], marker=".",linestyle='None',color='g')
# ax.plot(ans_point[0],ans_point[1],ans_point[2], marker="o",linestyle='None',color='r')
# ax.plot(ans2_point[0],ans2_point[1],ans2_point[2], marker="*",linestyle='None',color='b')

# plt.show()




if __name__ == "__main__":


    Points,vPoints_raw = Points_Generator()
    R,t = Rt_Generator(1,1,0)
    vPoints = (R@vPoints_raw.T + t).T

    vmap = Point_VoxelMap(Points)
    R_ans,t_ans = vmap.Matching(vPoints)

    voxels = vmap.Voxels_array
   

    print(R_ans)

    print(np.shape(voxels)[0])
    print(np.shape(voxels)[1])


    


    voxel_musx = np.zeros([np.shape(voxels)[0]*np.shape(voxels)[1],1])
    voxel_musy = np.zeros([np.shape(voxels)[0]*np.shape(voxels)[1],1])

    voxel_sigsx = np.zeros([4*np.shape(voxels)[0]*np.shape(voxels)[1],1])
    voxel_sigsy = np.zeros([4*np.shape(voxels)[0]*np.shape(voxels)[1],1])


    for i in range(0,np.shape(voxels)[0]):
        for j in range(0,np.shape(voxels)[1]):

            if voxels[i][j].empty == 0:
                print(i*np.shape(voxels)[1] + j)
                print(voxels[i][j].mu)
                voxel_musx[i*np.shape(voxels)[1] + j,0] = voxels[i][j].mu[0]
                voxel_musy[i*np.shape(voxels)[1] + j,0] = voxels[i][j].mu[1]


                voxel_sigsx[4*i*np.shape(voxels)[1] +j ,0] = voxels[i][j].mu[0] + np.sqrt(voxels[i][j].Sigma[0,0])
                voxel_sigsx[4*i*np.shape(voxels)[1] +j+1 ,0]= voxels[i][j].mu[0] 
                voxel_sigsx[4*i*np.shape(voxels)[1] +j+2 ,0]= voxels[i][j].mu[0] - np.sqrt(voxels[i][j].Sigma[0,0])
                voxel_sigsx[4*i*np.shape(voxels)[1] +j+3 ,0]= voxels[i][j].mu[0]


                voxel_sigsy[4*i*np.shape(voxels)[1] +j ,0] = voxels[i][j].mu[1] 
                voxel_sigsy[4*i*np.shape(voxels)[1] +j+1 ,0]= voxels[i][j].mu[1] - np.sqrt(voxels[i][j].Sigma[1,1])
                voxel_sigsy[4*i*np.shape(voxels)[1] +j+2 ,0]= voxels[i][j].mu[1] 
                voxel_sigsy[4*i*np.shape(voxels)[1] +j+3 ,0]= voxels[i][j].mu[1] + np.sqrt(voxels[i][j].Sigma[1,1])



    # kdmap = Point_Map(Points)
    # R_ans,t_ans = kdmap.Matching(vPoints)
    print(R_ans)

    print(R_ans.shape)
    print(t_ans.shape)
    vPoints_ans = (R_ans@vPoints.T + t_ans).T

    vPoints_ans2 = (R_ans.T@vPoints.T + t_ans).T

    vPoints_ans3 = (R_ans.T@vPoints.T - t_ans).T

    #Visualize
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot(Points[:,0],Points[:,1],marker=".",linestyle='None',color='b')
    ax.plot(vPoints[:,0],vPoints[:,1], marker=".",linestyle='None',color='g')
    ax.plot(vPoints_raw[:,0],vPoints_raw[:,1], marker=".",linestyle='None',color='b')
    ax.plot(vPoints_ans[:,0],vPoints_ans[:,1], marker=".",linestyle='None',color='r')
    # ax.plot(voxel_musx[:,0],voxel_musy[:,0], marker="*",linestyle='None',color='m')
    # ax.plot(voxel_sigsx[:,0],voxel_sigsy[:,0], marker=".",linestyle='-',color='m')
    # ax.plot(vPoints_ans3[:,0],vPoints_ans3[:,1], marker=".",linestyle='None',color='k')
  

    plt.show()
