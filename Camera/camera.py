import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import open3d as o3d

import cameraclass as cs
import mapclass as ms



# カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)

fast = cv2.FastFeatureDetector_create()

CAMERAHEIGHT = 480
CAMERAWIDTH = 640
INITLAYER = 2
INITSCALE = np.power(0.5,INITLAYER)
LAYERNUM = 4

MATCHING_DISTANCE_SCALE = 0.1

NoneType=type(None)

akaze = cv2.AKAZE_create()


DISTORTION = np.array([[ 0.11091643, -0.13274917, -0.0025602,  -0.00115712, -0.06822208]])


EPIPDISTTHR = 50

# elecom web camera  対角画角68度　　画素　480x640
K = np.zeros([3,3])
K[0, 0] = 593
K[1, 1] = 593
K[0, 2] = 320
K[1, 2] = 240
K[2, 2] = 1

Ks = []

for i in range(LAYERNUM):
    scale = np.power(0.5,i)
    Kc = K
    Kc[0:2,0:3] = scale*K[0:2,0:3]
    Ks.append(Kc)


INK = np.zeros([3, 3])
INK[0, 0] = 593*INITSCALE
INK[1, 1] = 593*INITSCALE
INK[0, 2] = 320*INITSCALE
INK[1, 2] = 240*INITSCALE
INK[2, 2] = 1


#Map
Map = ms.SLAMMap(LAYERNUM)
# initialize用のkeyframe
key_frame = cs.KeyFrame()


acounter = 100
key_frame_found = 0
initialized = 0

initialize_try_count = 0

# Viewer
# fig = plt.figure(figsize=(5, 5), dpi=120)
# ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect((1, 1, 1))


# q1 = Rectangle((0, 0), 0.3, 0.3, color='red')
# ax.add_patch(q1)
# art3d.pathpatch_2d_to_3d(q1, z=0, zdir=[0, 0, 1])


first_vec = np.reshape(np.array([0,0,1,0]),(4,1))
first_pos = np.reshape(np.array([0,0,0,1]),(4,1))

PosMtx = np.eye(4)


while True:
    acounter = acounter + 1
    # カメラからの画像取得 480 * 640 画素を想定
    ret, camera_frameraw = cap.read()
    

    if key_frame_found ==1 and initialized ==1:    
        for el in range(LAYERNUM):
            # 小さい順
            l = LAYERNUM -el-1
            scale = np.power(0.5,l)
            camera_frame = cv2.resize(camera_frameraw, None, None,
                              scale, scale, cv2.INTER_NEAREST)
            cam_kp,cam_kpdes = akaze.detectAndCompute(camera_frame, None)
            
            key_kpdes = Map.MapPoints.layeredPoints[l].kpdes
            
            points = Map.MapPoints.layeredPoints[l].X
            
            print(np.array((cam_kpdes)).shape)
            print("key")
            print(np.array(key_kpdes).shape)
            
           
            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            matches = bf.match(cam_kpdes,key_kpdes)
            atches = sorted(matches, key=lambda x: x.distance)
            
            cam_kp_ex = []
            key_kp_3D = []
            count = 0
                        
            for match in  matches:
                keyid = match.trainIdx
                matchid = match.queryIdx
                count += 1
                
                key_kp_3D.append(points[keyid])
                cam_kp_ex.append(cam_kp[matchid].pt)
                
                
            if count < 6:
                continue
    
            else:
                cam_kp_ex = cv2.undistortPoints(np.expand_dims(
                cam_kp_ex, axis=1), cameraMatrix=Ks[l], distCoeffs=DISTORTION,P = Ks[l])
                # これよくよく考えたら意味ないから直したいです    
                mask = cs.EpipolarCheck(key_kp_3D,cam_kp_ex,PosMtx,Ks[l],EPIPDISTTHR*scale)
                key_kp_3D = key_kp_3D[mask.ravel()==1]
                cam_kp_ex = cam_kp_ex[mask.ravel()==1]
                
                transpoints = np.linalg.inv(PosMtx)@key_kp_3D
                updatePose = cs.GaussNewtonOptimizer(transpoints,cam_kp_ex,Ks[l])
                
                PosMtx = PosMtx@updatePose

    
    elif acounter == 99 and key_frame_found ==1 and initialized ==0 :
        camera_frame = cv2.resize(camera_frameraw, None, None,
                              INITSCALE, INITSCALE, cv2.INTER_NEAREST)
        cam_kp,cam_kpdes = akaze.detectAndCompute(camera_frame, None)
        key_kp = Map.KeyFrames[0].layeredImage[INITLAYER].kp
        key_kpdes = Map.KeyFrames[0].layeredImage[INITLAYER].kpdes
        # queindex, matchindex, matchvalue = cs.AllMatching(camera_frame, cam_kp, key_frame, 9,MATCHING_DISTANCE_SCALE*CAMERAWIDTH*INITSCALE)

        cam_kp_ex = []
        key_kp_ex = []
        cam_kp_drw = []
        key_kp_drw = []
        key_kp_id = []
        count = 0
        
        # 反転防止　画像を9分割して　それぞれに特徴点があるようにする
        cam_height = np.ceil(float(camera_frame.shape[0])/3.)
        cam_width = np.ceil(float(camera_frame.shape[1])/3.)        
        arealist = np.zeros([3,3])
        
        # Brute-Force Matcher生成
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

        # 特徴量ベクトル同士をBrute-Force
        matches = bf.match(cam_kpdes,key_kpdes)
        atches = sorted(matches, key=lambda x: x.distance)
        
        #  opnecv の座標系は　(x:横軸  y:縦軸)   めんどくさいので　(x:縦軸  y:横軸) に変換したいが
        print(len(key_kpdes))
        print("camera total point num %d" % len(cam_kpdes))

        for match in  matches:
            keyid = match.trainIdx
            matchid = match.queryIdx
            
            count += 1
            if count < 20:
        #      if count < 20:
                key_kp_ex.append(key_kp[keyid].pt)
                key_kp_id.append(keyid)
                cam_kp_ex.append(cam_kp[matchid].pt)
                key_kp_drw.append(key_kp[keyid])
                cam_kp_drw.append(cam_kp[matchid])
                pt_areah = int(cam_kp[matchid].pt[1]//cam_height)
                pt_areaw = int(cam_kp[matchid].pt[0]//cam_width)
                
                arealist[pt_areah,pt_areaw] = 1
            # elif  np.sum(arealist) < 4:
            #     pt_areah = int(cam_kp[matchid].pt[1]//cam_height)
            #     pt_areaw = int(cam_kp[matchid].pt[0]//cam_width)
                
            #     if arealist[pt_areah,pt_areaw] == 0:
            #         key_kp_ex.append(key_kp[keyid].pt)
            #         cam_kp_ex.append(cam_kp[matchid].pt)
            #         key_kp_drw.append(key_kp[keyid])
            #         cam_kp_drw.append(cam_kp[matchid])                
            #         arealist[pt_areah,pt_areaw] = 1

            else:
                break
            
        print("valid camera num %d" % count)
        print("valid camera num %d" % int(np.sum(arealist)))
        
        
        
        if (count <= 5 or  np.sum(arealist) <3):
        # if (count <= 5):
            print("try again")
            initialize_try_count+=1
            
            if initialize_try_count > 3:
                print("key frame reset")
                initialize_try_count =0
                key_frame_found = 0
            #    reset
                Map = ms.SLAMMap(LAYERNUM)
                PosMtx = np.eye(4)
                
        else:
            view_frame3 = cv2.drawMatches(camera_frame, cam_kp,Map.KeyFrames[0].layeredImage[INITLAYER].image, key_kp, matches[:count-1], None, flags=2)
            cv2.imshow('camera3', view_frame3)
            
            cam_kp_ex = np.float32(cam_kp_ex)
            key_kp_ex = np.float32(key_kp_ex)
            

            
            # 3次元変換する前に必ず歪みは補正する
            cam_kp_ex = cv2.undistortPoints(np.expand_dims(
                cam_kp_ex, axis=1), cameraMatrix=INK, distCoeffs=DISTORTION,P = INK)
            key_kp_ex= cv2.undistortPoints(np.expand_dims(
                key_kp_ex, axis=1), cameraMatrix=INK, distCoeffs=DISTORTION,P = INK)
            
            
            # 5点アルゴリズム  Eを求める   keyframe - > current camera 
            # E, mask = cv2.findEssentialMat(key_kp_ex,cam_kp_ex, focal=INK[0, 0], pp=(
            #     INK[0, 2],INK[1, 2]), method=cv2.RANSAC, prob=0.999, threshold=3.0)
            
            F, mask = cv2.findFundamentalMat(key_kp_ex,cam_kp_ex,cv2.FM_LMEDS)
            
            E = INK.T@F@INK
            #mask はinliner outlinerのflg   
            key_kp_ex = key_kp_ex[mask.ravel()==1]
            cam_kp_ex = cam_kp_ex[mask.ravel()==1]
            key_kp_id = np.array(key_kp_id)
            key_kp_id = key_kp_id[mask.ravel()==1]

            #正しそうな回転　並進行列を求める 
            points, R, t, mask = cv2.recoverPose(E,key_kp_ex, cam_kp_ex,focal=INK[0, 0], pp=(
                INK[0, 2],INK[1, 2]))

            #  camera1座標系→camera2座標系の変換行列が出るので　姿勢変換行列としてはその逆行列
            R = np.linalg.inv(R)
            
            t = -R@t



            pn = len(key_kp_ex) 

            view_frame = camera_frame
            view_frame2 = Map.KeyFrames[0].layeredImage[INITLAYER].image
            
            
            
            
            for i in range(pn):
                view_frame = cv2.drawKeypoints(
                    view_frame, [cam_kp_drw[i]], None, color=(int(255*(pn-i)/pn), 0,int(255*i/pn)))
                view_frame2 = cv2.drawKeypoints(
                    view_frame2, [key_kp_drw[i]], None, color=(int(255*(pn-i)/pn),0, int(255*i/pn)))

            cv2.imshow('camera', view_frame)
            cv2.imshow('camera2', view_frame2)
            
            
            
            qq = np.reshape(np.array([0, 0, 1]),(3,1))

            qv = R@qq
            
            print("update rotation:")
            print(qv)
            print("update transition:")
            print(t)
            
            
            RtMtx = np.eye(4)
            for ki in range(3):
                for kj in range(3):
                    RtMtx[ki,kj]  = R[ki,kj]
                    
            for ki in range(3):
                RtMtx[ki,3] = t[ki]
            
            # 姿勢変換行列
            PosMtx = PosMtx@RtMtx

            # 有効判定②　姿勢角が変だったらやり直し 
            if np.abs(qv[2]) < 0.1  or qv[2] < 0.:
                print("invalid motion estimation. try again")
                initialize_try_count+=1
                if initialize_try_count > 3:
                    print("key frame reset")
                    initialize_try_count =0
                    key_frame_found = 0
                                    #    reset
                    Map = ms.SLAMMap(LAYERNUM)
                    PosMtx = np.eye(4)

                    
                continue
                
            if initialized == 0:
                initialized = 1


                Map.SetKeyFrame(camera_frameraw,PosMtx)
                mappoints,ori,pts1_2img= cs.Get3DPoint(key_kp_ex,cam_kp_ex,PosMtx,INK)
                
                
                print(mappoints)
                
                Map.SetMapPoints(mappoints.reshape([-1,3,1]),INITLAYER,0,key_kp_id)
                
                
            points = []
            colors = []
            print(mappoints)
            for ip in range(mappoints.shape[0]):
            
                        points.append([mappoints[ip,0,0],mappoints[ip,1,0],mappoints[ip,2,0]])


            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)


            # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, -0])
            
            # mesh_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[[0],t[1],t[2]])
            
            # mesh_frame2.rotate(R, center=([0],t[1],t[2]))
            
            

            
            print(pts1_2img.shape)                
            for pp in range(pts1_2img.shape[0]):          
                cv2.line(view_frame, (int(ori[0]), int(ori[1])), (int(pts1_2img[pp,0,0]), int(pts1_2img[pp,0,1])), (0, 200, 0), 1)

            cv2.imshow('camera', view_frame)
            # o3d.visualization.draw_geometries([pcd,mesh_frame,mesh_frame2])
    elif acounter >= 100:
            acounter = 0      
            if key_frame_found == 0:
                camera_frame = cv2.resize(camera_frameraw, None, None,
                              INITSCALE, INITSCALE, cv2.INTER_NEAREST)
                key_frame.SetImage(camera_frame)
                kp,des = akaze.detectAndCompute(key_frame.image, None)
                print(len(kp))
                if len(kp) > 10 :
                    key_frame_found = 1
                    Map.SetKeyFrame(camera_frameraw,PosMtx)
                    print("SetKeyFrame")
                    

                    view_frame2 = cv2.drawKeypoints(camera_frame, kp,None,color=(0,255,0))
                    cv2.imshow('camera2', view_frame2)


    # 繰り返し分から抜けるためのif文
    key = cv2.waitKey(10)
    if key == 27:
        break

# メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()
