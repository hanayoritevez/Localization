import numpy as np
import cv2
from matplotlib import pyplot as plt

#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)

fast = cv2.FastFeatureDetector_create()
while True:
    #カメラからの画像取得
    ret, camera_frame = cap.read()
        # まずデフォルト値で FASTオブジェクトを作る
    
    camera_frame2 = cv2.resize(camera_frame, None, None, 0.5, 0.5, cv2.INTER_NEAREST)
    camera_frame3 = cv2.resize(camera_frame, None, None, 0.25, 0.25, cv2.INTER_NEAREST)
    camera_frame4 = cv2.resize(camera_frame, None, None, 0.125, 0.125, cv2.INTER_NEAREST)
    
    keypoints = fast.detect(camera_frame,None)
    keypoints2 = fast.detect(camera_frame2,None)
    
    print("num of keypoints in each layer")
    print(len(keypoints))
    print(keypoints[0].pt[0])
    print(len(keypoints2))
    

    # for i, point in enumerate(keypoints2):
    #     print(point)
    #     keypoints2[i].pt[0] = 2*keypoints2[i].pt[0]
    #     keypoints2[i].pt[1] = 2*keypoints2[i].pt[1]

    keypoints3 = fast.detect(camera_frame3,None)
    
    print(len(keypoints3))
    # for point in enumerate(keypoints3):
    #     point.pt[0] = 4*point.pt[0]
    #     point.pt[1] = 4*point.pt[1]

    keypoints4 = fast.detect(camera_frame4,None)
    print(len(keypoints4))
    # for point in enumerate(keypoints4):
    #     point.pt[0] = 8*point.pt[0]
    #     point.pt[1] = 8*point.pt[1]

    view_frame = cv2.drawKeypoints(camera_frame, keypoints,None,color=(255,0,0))
    
    view_frame2 = cv2.drawKeypoints(camera_frame2, keypoints2,None,color=(0,255,0))
    view_frame3 = cv2.drawKeypoints(camera_frame3, keypoints3,None,color=(0,0,255))
    view_frame4 = cv2.drawKeypoints(camera_frame4, keypoints4,None,color=(0,255,255))
    

    #カメラの画像の出力
    cv2.imshow('camera' , view_frame)
    cv2.imshow('camera2' , view_frame2)
    cv2.imshow('camera3' , view_frame3)
    cv2.imshow('camera4' , view_frame4)

    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(10)
    if key == 27:
        break

#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()