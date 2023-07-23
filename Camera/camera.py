import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import cameraclass as cs

# カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)

fast = cv2.FastFeatureDetector_create()


Scale = 0.5

# elecom web camera  対角画角68度　　画素　480x640
K = np.zeros([3, 3])
K[0, 0] = 593*Scale
K[1, 1] = 593*Scale
K[0, 2] = 240*Scale
K[1, 2] = 320*Scale
K[2, 2] = 1




key_frame = cs.KeyFrame()

acounter = 20
key_frame_found = 0


# Viewer
fig = plt.figure(figsize=(5, 5), dpi=120)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1))


q1 = Rectangle((0, 0), 0.3, 0.3, color='red')
ax.add_patch(q1)
art3d.pathpatch_2d_to_3d(q1, z=0, zdir=[0, 0, 1])

USEAKAZE = 0
while True:

    acounter = acounter + 1
    # カメラからの画像取得 480 * 640 画素を想定
    ret, camera_frame = cap.read()

    
    
    
    camera_frame = cv2.resize(camera_frame, None, None,
                              Scale, Scale, cv2.INTER_NEAREST)

    if acounter == 19:

        if USEAKAZE:
            detector = cv2.AKAZE_create()
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            cam_kpA, cam_desA = detector.detectAndCompute(camera_frame, None)
            key_kpA, key_desA = detector.detectAndCompute(
                key_frame.image, None)

            good = []
            pts1 = []
            pts2 = []
            
            print(cam_kpA)
            print(key_kpA)
            
            
            

            matches = bf.match(cam_desA,key_desA)

            matches = sorted(matches, key=lambda x: x.distance)

            count = 0
            for m in matches:
                count += 1
                if count < 6:
                    good.append([m])
                    pts2.append(key_kpA[m.trainIdx].pt)
                    pts1.append(cam_kpA[m.queryIdx].pt)
                    
    

            pts1 = np.float32(pts1)
            pts2 = np.float32(pts2)
            pts1_norm = cv2.undistortPoints(np.expand_dims(
                pts1, axis=1), cameraMatrix=K, distCoeffs=None)
            pts2_norm = cv2.undistortPoints(np.expand_dims(
                pts2, axis=1), cameraMatrix=K, distCoeffs=None)

            E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(
                0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
            points, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
            
            view_frame = cv2.drawKeypoints(
                camera_frame, cam_kpA, None, color=(255, 0, 0))
            view_frame2 = cv2.drawKeypoints(
                key_frame.image, key_kpA, None, color=(255, 0, 0))

            cv2.imshow('camera', view_frame)
            cv2.imshow('camera2', view_frame2)

            qq = np.array([0, 0, 1])

            qv = R@qq.T

            print(qv)

            print(t)

        else:
            cam_kp = fast.detect(camera_frame, None)

            key_kp = key_frame.kp
            queindex, matchindex, matchvalue = cs.AllMatching(camera_frame, cam_kp, key_frame, 9,20)

            cam_kp_ex = []
            key_kp_ex = []
            cam_kp_drw = []
            key_kp_drw = []
            count = 0
            for queind, matchind in zip(queindex, matchindex):
                count += 1
                if count < 20:

                    key_kp_ex.append(key_kp[queind].pt)
                    cam_kp_ex.append(cam_kp[matchind].pt)
                    key_kp_drw.append(key_kp[queind])
                    cam_kp_drw.append(cam_kp[matchind])
                else:
                    break

            cam_kp_ex = np.float32(cam_kp_ex)
            key_kp_ex = np.float32(key_kp_ex)

            cam_kp_norm = cv2.undistortPoints(np.expand_dims(
                cam_kp_ex, axis=1), cameraMatrix=K, distCoeffs=None)
            key_kp_norm = cv2.undistortPoints(np.expand_dims(
                key_kp_ex, axis=1), cameraMatrix=K, distCoeffs=None)

            # 5点アルゴリズム
            E, mask = cv2.findEssentialMat(cam_kp_ex, key_kp_ex, focal=593, pp=(
                240., 320.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
            points, R, t, mask = cv2.recoverPose(E, cam_kp_norm, key_kp_norm)
            
            
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

            qq = np.array([0, 0, 1])

            qv = R@qq.T

            print(qv)

            print(t)

            # q2 = Rectangle((t[0], t[1]), 0.3, 0.3, color='blue')
            # ax.add_patch(q2)
            # art3d.pathpatch_2d_to_3d(q2, z=t[2], zdir=qv)

            # plt.show(block=False)

    if acounter >= 20:
            acounter = 0

            if key_frame_found == 0:
                key_frame.SetImage(camera_frame)
                kp = fast.detect(key_frame.image, None)
                key_frame.SetKp(kp)
                key_frame_found = 1
                
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
