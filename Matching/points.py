
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
        
        
        self.Points = np.empty([0,3])
        self.mu =None
        self.Sigma_inv = None
        self.empty = 1
        self.dimension = np.array([1.5 ,1.5 ,1.5])

        if Points:
            
            self.Points = Points
            self.mu = np.average(self.Points,axis=0)
         
            if self.Points.shape[0] < 5:
                self.Sigma_inv = np.diag((self.dimension/2.)*(self.dimension/2.))
            else:
                self.Sigma_inv = np.linalg.inv(np.cov(self.Points.T))
                self.empty =0

    def add_point(self,point):
        
        self.Points = np.concatenate([self.Points,np.reshape(point,[1,3])],0)
        self.mu = np.average(self.Points,axis = 0)
        
        if self.Points.shape[0] < 5:
            self.Sigma_inv = np.diag((self.dimension/2.)*(self.dimension/2.))
        else:
            # print(self.Points)
            # self.Sigma_inv = np.linalg.inv(np.cov(self.Points.T ) + 0.0001*np.eye(3))
            # print(self.Sigma_inv)

            self.Sigma_inv = np.linalg.inv(np.cov(self.Points.T ) )


            
            
            self.empty =0



class Point_VoxelMap:
    def __init__(self,mPoints,voxel_size = [5,5,5]):
        
        max_x = [-100000,-100000,-1000000]
        min_x = [100000,100000,1000000]
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
                zs = []
                for k in range(self.Voxels_size[2]):
                    zs.append(Point_Voxel())
                ys.append(zs)            
            self.Voxels_array.append(ys)

        voxel_indexs =np.apply_along_axis(self.search_voxel,1,mPoints)
        for index,X in zip(voxel_indexs,mPoints):
            
            self.Voxels(index).add_point(X)


        self.d1 = 1
        self.d2 = 1


    def Matching(self,vPoints):
        P = np.array([[0.],[0.],[0.],[0.],[0.],[0.]])
        last_P = np.array([[0.],[0.],[0.],[0.],[0.],[0.]])
        pre_score = -100.
        score = -100.

        counter = 0
        # while(pre_score < score):
        while(counter<100):
            pre_score = score
            trans_vPoints = (self.step_R(P)@vPoints.T+ np.reshape(P[0:3,0],[3,1])).T
            voxel_indexs =np.apply_along_axis(self.search_voxel,1,trans_vPoints).tolist()
          
            P,score = self.ndt_matching(trans_vPoints,voxel_indexs,P)

            counter = counter + 1


            # if pre_score  > score:
            #     break

        return self.step_R(P),np.reshape(P[0:3,0],[3,1])


    def Voxels(self,index):

        return self.Voxels_array[index[0]][index[1]][index[2]]

            
    
    def search_voxel(self,X):
        dindexes = []
        for i in range(3):
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

        Hs = np.zeros([6,6])
        gs = np.zeros([6,1])
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



        if voxel_index[0] <0    or  voxel_index[1] <0 or voxel_index[2] <0:
                
                return -0


        if self.Voxels(voxel_index).empty == 1:
            return -0


        
      
        
             
        score = -self.d1*np.exp(-self.d2/2. *(X_k -self.Voxels(voxel_index).mu).T@self.Voxels(voxel_index).Sigma_inv@(X_k -self.Voxels(voxel_index).mu))

        return score
        

    def step_R(self,P):


        p_rx = P[3,0]
        p_ry = P[4,0]
        p_rz = P[5,0]
        
        cx = np.cos(p_rx)
        sx = np.sin(p_rx)
        cy = np.cos(p_ry)
        sy = np.sin(p_ry)
        cz = np.cos(p_rz)
        sz = np.sin(p_rz)


        R = np.zeros([3,3])

        R[0,0] =  cy*cz
        R[0,1] = -cy*sz
        R[0,2] = sy
        R[1,0] = cx*sz + sx*sy*cz
        R[1,1] = cx*cz -sx*sy*sz 
        R[1,2] = -sx*cy
        R[2,0] = sx*sz -cx*sy*cz
        R[2,1] = cx*sy*sz +sx*cz
        R[2,2] = cx*cy


        return R


    def  calc_Hg(self,X_k_raw,voxel_index,P):

        
        Hk = np.zeros([6,6])

        gk = np.zeros([6,1])


        if voxel_index[0] < 0 or voxel_index[1] < 0 or voxel_index[2] < 0:
            return Hk,gk
        if self.Voxels(voxel_index).empty == 1:

            return Hk,gk


        

        X_k = np.reshape(X_k_raw -self.Voxels(voxel_index).mu,[3,1])

        Dt = X_k.T@self.Voxels(voxel_index).Sigma_inv
        Maht = X_k.T@self.Voxels(voxel_index).Sigma_inv@X_k

        if Maht > 1.:
            Dt = X_k.T@self.Voxels(voxel_index).Sigma_inv/np.sqrt(Maht)*0.9
            X_k = X_k/np.sqrt(Maht)*0.9

        if  self.d2*np.exp(-self.d2/2.*Dt@X_k) >1 or self.d2*np.exp(-self.d2/2.*Dt@X_k) <0:


            return Hk,gk
        
        
        Sigma_k_inv = self.Voxels(voxel_index).Sigma_inv
        Ht = self.calc_hessian(np.reshape(X_k_raw,[3,1]),P)
        Jt = self.calc_jacobian(np.reshape(X_k_raw,[3,1]),P)
        
        gk  =  (np.exp(-self.d2/2.*Dt@X_k)*self.d1*self.d2*Dt@Jt).T

        for i in range(6):
            for j in range(6):
                Hk[i,j] = self.d1*self.d2*np.exp(-self.d2/2.*Dt@X_k)*(-self.d2* (Dt@Jt[:,i])*(Dt@Jt[:,j]) + Dt@Ht[:,i,j] + Jt[:,j].T@Sigma_k_inv@Jt[:,i])
        
        return Hk,gk
        


    def calc_jacobian(self, X,P):
        
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        p_rx = P[3,0]
        p_ry = P[4,0]
        p_rz = P[5,0]
        
        cx = np.cos(p_rx)
        sx = np.sin(p_rx)
        cy = np.cos(p_ry)
        sy = np.sin(p_ry)
        cz = np.cos(p_rz)
        sz = np.sin(p_rz)

        Ja = x*(-sx*sz + cx*sy*cz) + y*(-sx*cz -cx*sy*sz) + z*(-cx*cy)
        Jb = x*(cx*sz + sx*sy*cz) + y*(-sx*sy*sz + cx*cz) + z*(-sx*cy)
        Jc = x*(-sy*cz) + y*(sy*sz) + z*(cy)
        Jd = x*(sx*cy*cz) + y*(-sx*cy*sz) + z*(sx*sy)
        Je = x*(-cx*cy*cz) + y*(cx*cy*sz) + z*(-cx*sy)
        Jf = x*(-cy*sz) + y*(-cy*cz)
        Jg = x*(cx*cz - sx*sy*sz) + y*(-cx*sz - sx*sy*cz)
        Jh = x*(sx*cz + cx*sy*sz) + y*(cx*sy*cz -sx*sz)

        J = np.zeros([3,6])
        J[0,0] = 1.
        J[1,1] = 1.
        J[2,2] = 1.
        J[0,4] = Jc #z
        J[0,5] = Jf #-y
        J[1,3] = Ja #-z
        J[1,4] = Jd #0
        J[1,5] = Jg #x
        J[2,3] = Jb #y
        J[2,4] = Je #-x
        J[2,5] = Jh #0
        return J 

    def calc_hessian(self, X,P):

        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        p_rx = P[3,0]
        p_ry = P[4,0]
        p_rz = P[5,0]

        cx = np.cos(p_rx)
        sx = np.sin(p_rx)
        cy = np.cos(p_ry)
        sy = np.sin(p_ry)
        cz = np.cos(p_rz)
        sz = np.sin(p_rz)

        Hax = 0.
        Hay = x*(-cx*cz - sx*sy*cz) + y*(-cx*cz + sx*sy*sz) + z*(sx*cy)
        Haz = x*(-sx*sz + cx*sy*cz) + y*(-cx*sy*sz -sx*cz) + z*(-cx*cy)
        Hbx = 0.
        Hby = x*(cx*cy*cz) + y*(-cx*cy*sz) + z*(cx*sy)
        Hbz = x*(sx*cy*cz) + y*(-sx*cy*sz) + z*(sx*sy)
        Hcx = 0.
        Hcy = x*(-sx*cz - cx*sy*sz) + y*(sx*sz -cx*sy*cz)
        Hcz = x*(cx*cz - sx*sy*sz) + y*(-cx*sz -sx*sy*cz)
        Hdx = x*(-cy*cz) + y*(cy*sz) + z*(-sy)
        Hdy = x*(-sx*sy*cz) + y*(sx*sy*sz) + z*(sx*cy)
        Hdz = x*(cx*sy*cz) + y*(-cx*sy*sz) + z*(-cx*cy)
        Hex = x*(sy*sz) + y*(sy*cz)
        Hey = x*(-sx*cy*sz) + y*(-sx*cy*cz)
        Hez = x*(cx*cy*sz) + y*(cx*cy*cz)
        Hfx = x*(-cy*cz) + y*(cy*sz)
        Hfy = x*(-cx*sz - sx*sy*cz) + y*(-cx*cz + sx*sy*sz)
        Hfz = x*(-sx*sz + cx*sy*cz) + y*(-cx*sy*sz - sx*cz)
        H = np.zeros([3,6,6])

        H[0,3,3] = Hax
        H[1,3,3] = Hay
        H[2,3,3] = Haz

        H[0,3,4] = Hbx
        H[1,3,4] = Hby
        H[2,3,4] = Hbz

        H[0,3,5] = Hcx
        H[1,3,5] = Hcy
        H[2,3,5] = Hcz

        H[0,4,3] = Hbx
        H[1,4,3] = Hby
        H[2,4,3] = Hbz

        H[0,4,4] = Hdx
        H[1,4,4] = Hdy
        H[2,4,4] = Hdz

        H[0,4,5] = Hex
        H[1,4,5] = Hey
        H[2,4,5] = Hez

        H[0,5,3] = Hcx
        H[1,5,3] = Hcy
        H[2,5,3] = Hcz 

        H[0,5,4] = Hex
        H[1,5,4] = Hey
        H[2,5,4] = Hez

        H[0,5,5] = Hfx
        H[1,5,5] = Hfy
        H[2,5,5] = Hfz


        return H

def Points_Generator():


    
    Points_xy = 10*np.random.rand(1000,3)
    Points_xy[:,2] = 0.01*Points_xy[:,2]
    Points_xz1 = 10.*np.random.rand(1000,3)
    Points_xz1[:,1] = 0.01*Points_xz1[:,1]
    Points_xz2 = 10*np.random.rand(1000,3)
    Points_xz2[:,1] = 10.+ 0.01*Points_xz2[:,1]
    Points_yz = 10*np.random.rand(1000,3)
    Points_yz[:,0] = 10.+ 0.01*Points_yz[:,0]


    Points = np.concatenate([Points_xz1,Points_xz2,Points_yz])

    vPoints_xy = 10*np.random.rand(200,3)
    vPoints_xy[:,2] = 0.01*vPoints_xy[:,2]
    vPoints_xz1 = 10.*np.random.rand(200,3)
    vPoints_xz1[:,1] = 0.01*vPoints_xz1[:,1]
    vPoints_xz2 = 10*np.random.rand(200,3)
    vPoints_xz2[:,1] = 10.+ 0.01*vPoints_xz2[:,1]
    vPoints_yz = 10*np.random.rand(200,3)
    vPoints_yz[:,0] = 10.+ 0.01*vPoints_yz[:,0]
    vPoints = np.concatenate([vPoints_xz1,vPoints_xz2,vPoints_yz])
    for i in range(vPoints.shape[0]):
        vPoints[i] = vPoints[i] + np.random.randn(1,3)*0.1
    print(vPoints)

    return Points,vPoints 




def Rt_Generator(bx = 1,by=1,bz = 0):



    T = np.random.rand(3,1)*0.5

    T[0] = T[0]*bx
    T[1] = T[1]*by
    T[2] = T[2]*bz

    r = np.arcsin(np.random.rand(3,1)*0.5)

    Rx = np.eye(3)
    Ry = np.eye(3)
    Rz = np.eye(3)

    theta_x = r[0]*by*bz
    theta_y = r[1]*bx*bz
    theta_z = r[2]*by*bx

    Rx[1,1] = np.cos(theta_x)
    Rx[2,2] = np.cos(theta_x)
    Rx[1,2] = -np.sin(theta_x)
    Rx[2,1] = np.sin(theta_x)
    Ry[0,0] = np.cos(theta_y)
    Ry[2,2] = np.cos(theta_y)
    Ry[0,2] = np.sin(theta_y)
    Ry[2,0] = -np.sin(theta_y)
    Rz[0,0] = np.cos(theta_z)
    Rz[1,1] = np.cos(theta_z)
    Rz[0,1] = -np.sin(theta_z)
    Rz[1,0] = np.sin(theta_z)

    print(T)
    print(r)


    return Rx@Ry@Rz,T




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
    print(R_ans)

    # kdmap = Point_Map(Points)
    # R_ans,t_ans = kdmap.Matching(vPoints)
    # print(R_ans)

    print(R_ans.shape)
    print(t_ans.shape)
    vPoints_ans = (R_ans@vPoints.T + t_ans).T

    #Visualize
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot(Points[:,0],Points[:,1],Points[:,2],marker=".",linestyle='None',color='b')
    ax.plot(vPoints[:,0],vPoints[:,1],vPoints[:,2], marker=".",linestyle='None',color='g')
    ax.plot(vPoints_raw[:,0],vPoints_raw[:,1],vPoints_raw[:,2], marker=".",linestyle='None',color='b')
    ax.plot(vPoints_ans[:,0],vPoints_ans[:,1],vPoints_ans[:,2], marker=".",linestyle='None',color='r')
  

    plt.show()
