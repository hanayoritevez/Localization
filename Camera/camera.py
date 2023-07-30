import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import cameraclass as cs
import mapclass as ms

# カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)

fast = cv2.FastFeatureDetector_create()

CAMERAHEIGHT = 480
CAMERAWIDTH = 640
INITSCALE = 0.5

# elecom web camera  対角画角68度　　画素　480x640
K = np.zeros([3,3])
K[0, 0] = 593
K[1, 1] = 593
K[0, 2] = 240
K[1, 2] = 320
K[2, 2] = 1


INK = np.zeros([3, 3])
INK[0, 0] = 593*INITSCALE
INK[1, 1] = 593*INITSCALE
INK[0, 2] = 240*INITSCALE
INK[1, 2] = 320*INITSCALE
INK[2, 2] = 1

#Map
Map = ms.SLAMMap()
# initialize用のkeyframe
key_frame = cs.KeyFrame()


acounter = 20
key_frame_found = 0
initialized = 0

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
    


    camera_frame = cv2.resize(camera_frameraw, None, None,
                              INITSCALE, INITSCALE, cv2.INTER_NEAREST)

    if acounter == 19 and key_frame_found ==1:


            cam_kp = fast.detect(camera_frame, None)

            key_kp = key_frame.kp
            queindex, matchindex, matchvalue = cs.AllMatching(camera_frame, cam_kp, key_frame, 9,20)

            cam_kp_ex = []
            key_kp_ex = []
            cam_kp_drw = []
            key_kp_drw = []
            count = 0
            
            # 反転防止　画像を9分割して　それぞれに特徴点があるようにする
            cam_height = np.ceil(float(camera_frame.shape[0])/3.)
            cam_width = np.ceil(float(camera_frame.shape[1])/3.)
            
            arealist = np.zeros([3,3])
            
            for queind, matchind in zip(queindex, matchindex):
                count += 1
                if count < 20 or  np.sum(arealist) < 6 :
                    key_kp_ex.append(key_kp[queind].pt)
                    cam_kp_ex.append(cam_kp[matchind].pt)
                    key_kp_drw.append(key_kp[queind])
                    cam_kp_drw.append(cam_kp[matchind])
                    pt_areah = int(cam_kp[matchind].pt[0]//cam_height)
                    pt_areaw = int(cam_kp[matchind].pt[1]//cam_width)
                    
                    arealist[pt_areah,pt_areaw] = 1
                else:
                    break
                
            print(count)
            if  count > 5 and np.sum(arealist) >=6:
                cam_kp_ex = np.float32(cam_kp_ex)
                key_kp_ex = np.float32(key_kp_ex)
                
                
                
                

                # 5点アルゴリズム  Eを求める   keyframe - > current camera 
                E, mask = cv2.findEssentialMat(key_kp_ex,cam_kp_ex, focal=593, pp=(
                    CAMERAHEIGHT*INITSCALE, CAMERAWIDTH*INITSCALE), method=cv2.RANSAC, prob=0.999, threshold=3.0)
                #mask はinliner outlinerのflg
                
                key_kp_ex = key_kp_ex[mask.ravel()==1]
                cam_kp_ex = cam_kp_ex[mask.ravel()==1]
                print(len(key_kp_ex))
                points, R, t, mask = cv2.recoverPose(E, key_kp_ex,cam_kp_ex)
                # outliner取り除き



                
                # ゆがみなければ単に　z = 1の平面上
                cam_kp_norm = cv2.undistortPoints(np.expand_dims(
                    cam_kp_ex, axis=1), cameraMatrix=INK, distCoeffs=None)
                key_kp_norm = cv2.undistortPoints(np.expand_dims(
                    key_kp_ex, axis=1), cameraMatrix=INK, distCoeffs=None)
                
                
                pn = len(key_kp_ex) 
                # print(cam_kp_norm)
                # print(key_kp_norm)
                view_frame = camera_frame
                view_frame2 = key_frame.image
                
                for i in range(pn):
                    view_frame = cv2.drawKeypoints(
                        view_frame, [cam_kp_drw[i]], None, color=(int(255*(pn-i)/pn), 0,int(255*i/pn)))
                    view_frame2 = cv2.drawKeypoints(
                        view_frame2, [key_kp_drw[i]], None, color=(int(255*(pn-i)/pn),0, int(255*i/pn)))

                cv2.imshow('camera', view_frame)
                cv2.imshow('camera2', view_frame2)
                
                qq = np.reshape(np.array([0, 0, 1]),(3,1))

                qv = R@qq
                
                print("update rotation")
                print(qv)
                print("update transition")
                print(t)
                
                
                RtMtx = np.eye(4)
                
                for ki in range(3):
                    for kj in range(3):
                        RtMtx[ki,kj]  = R[ki,kj]
                        
                for ki in range(3):
                    RtMtx[ki,3] = t[ki]
                
                print("direction")
                print(PosMtx@RtMtx@first_vec)
                print("position")
                print(PosMtx@RtMtx@first_pos)
                
                
                PosMtx = PosMtx@RtMtx

                # q2 = Rectangle((t[0], t[1]), 0.3, 0.3, color='blue')
                # ax.add_patch(q2)
                # art3d.pathpatch_2d_to_3d(q2, z=t[2], zdir=qv)

                # plt.show(block=False)
                
                
                if initialized == 0:
                    initialized = 1
                    key_frame_p = cs.KeyFramePyramid(4)
                    key_frame_p.SetImage(camera_frameraw)
                    key_frame_p.SetPose(PosMtx)
                    Map.SetKeyFrame(key_frame_p)
                    mappoints = cs.Get3DPoint(key_kp_norm,cam_kp_ex,PosMtx,INK)
                    
                    print(mappoints)
                    # Map.SetPoints(key_frame_p)
                
                key_frame.SetImage(camera_frame)
                key_frame.SetKp(cam_kp)
                
                print("KeyFrame Update")
                
              
    # elif initialized ==1:
    #     # last movement applied and set initial pose
    #      print("cccccccccccccccccccccccccccccccccccccccc")
    #     # det
    elif acounter >= 20:
            acounter = 0      
            if key_frame_found == 0:
                key_frame.SetImage(camera_frame)
                kp = fast.detect(key_frame.image, None)
                key_frame.SetKp(kp)
                if len(kp) > 10 :
                    key_frame_found = 1
                    key_frame_p = cs.KeyFramePyramid(4)
                    key_frame_p.SetImage(camera_frameraw)
                    Map.SetKeyFrame(key_frame_p)
                    print("SetKeyFrame")

        # camera_frame2 = cv2.resize(camera_frame, None, None, 0.5, 0.5, cv2.INTER_NEAREST)
        # camera_frame3 = cv2.resize(camera_frame, None, None, 0.25, 0.25, cv2.INTER_NEAREST)
        # camera_frame4 = cv2.resize(camera_frame, None, None, 0.125, 0.125, cv2.INTER_NEAREST)

        # keypoints = fast.detect(camera_frame,None)
        # keypoints2 = fast.detect(camera_frame2,None)

        # print("num of keypoints in each layer")
        # print(len(keypoints))
        # print(keypoints[0].pt[0])
        # print(len(keypoints2))

        # for i, point in enumerate(keypoints2):
        #     print(point)
        #     keypoints2[i].pt[0] = 2*keypoints2[i].pt[0]
        #     keypoints2[i].pt[1] = 2*keypoints2[i].pt[1]

        # keypoints3 = fast.detect(camera_frame3,None)

        # print(len(keypoints3))
        # for point in enumerate(keypoints3):
        #     point.pt[0] = 4*point.pt[0]
        #     point.pt[1] = 4*point.pt[1]

        # keypoints4 = fast.detect(camera_frame4,None)
        # print(len(keypoints4))
        # for point in enumerate(keypoints4):
        #     point.pt[0] = 8*point.pt[0]
        #     point.pt[1] = 8*point.pt[1]

        # view_frame = cv2.drawKeypoints(camera_frame, keypoints,None,color=(255,0,0))
        # view_frame2 = cv2.drawKeypoints(camera_frame2, keypoints2,None,color=(0,255,0))
        # view_frame3 = cv2.drawKeypoints(camera_frame3, keypoints3,None,color=(0,0,255))
        # view_frame4 = cv2.drawKeypoints(camera_frame4, keypoints4,None,color=(0,255,255))

        # カメラの画像の出力
        # cv2.imshow('camera' , view_frame)
        # cv2.imshow('camera2' , view_frame2)
        # cv2.imshow('camera3' , view_frame3)
        # cv2.imshow('camera4' , view_frame4)

    # 繰り返し分から抜けるためのif文
    key = cv2.waitKey(10)
    if key == 27:
        break

# メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()
