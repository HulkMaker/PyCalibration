#coding:utf-8
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

dirs=('Left','Right')
LR_matrix=[[[0 for i in range(3)] for j in range(3)] for k in range(len(dirs))]
LR_distortion=[[[0 for i in range(5)] for j in range(1)] for k in range(len(dirs))]
images = glob.glob(dirs[0]+'/*.png')
LR_rotation=[[[0 for i in range(3)] for j in range(len(images))] for k in range(len(dirs))]
LR_transpose=[[[0 for i in range(3)] for j in range(len(images))] for k in range(len(dirs))]
for i in range(2):
    # 找棋盘格角点
    # 阈值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #棋盘格模板规格
    w = 9
    h = 6
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # 储存棋盘格角点的世界坐标和图像坐标对
    objpoints = [] # 在世界坐标系中的三维点
    imgpoints = [] # 在图像平面的二维点

    images = glob.glob(dirs[i]+'/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 找到棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
        # 如果找到足够点对，将其存储起来
        if ret== True:
            cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w,h), corners, ret)
            cv2.imshow('find corners in '+dirs[i],img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    # calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    #dist = np.array([-0.13615181, 0.53005398, 0, 0, 0]) # no translation
    #print "ret:", ret
    #print "mtx:\n", mtx  # 内参数矩阵
    #print "dist:\n", dist # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
    #print "rvecs:\n", rvecs # 旋转向量  # 外参数
    #print "tvecs:\n", tvecs  # 平移向量  # 外参数
    #print rvecs[0],"\n",rvecs[1],"\n",rvecs[2],"\n",rvecs[3]
    LR_matrix[i]=mtx
    LR_distortion[i]=dist
    LR_rotation[i]=rvecs
    LR_transpose[i] = tvecs
    print dirs[i],"intrinsic:\n ",LR_matrix[i]
    print LR_distortion[i]

# calculate extrinsic result
R=np.zeros(3)
T=np.zeros(3)
size = (332, 252) # 图像尺寸
#print LR_rotation[0][0],LR_transpose[0][0],LR_rotation[0][1],LR_transpose[0][1]
cv2.composeRT(LR_rotation[0][0],LR_transpose[0][0],LR_rotation[0][1],LR_transpose[0][1],R,T)
# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(LR_matrix[0], LR_distortion[0],LR_matrix[1], LR_distortion[1], size, LR_rotation[0][1],LR_transpose[0][1])
print "R1:\n",R1,"\nR2:\n",R2,"\n",Q

"""
img_L = cv2.imread("Ll.png", 0)
img_R = cv2.imread("R1.png", 0)

#stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, 16, 15)
stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)  #OpenCV 3.0的函数
disparity = stereo.compute(img_L, img_R)

plt.subplot(121), plt.imshow(img_L, 'gray'), plt.title('img_left'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(disparity, 'gray'), plt.title('disparity'), plt.xticks([]), plt.yticks([])
plt.show()
"""

