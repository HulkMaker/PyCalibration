#coding:utf-8
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt


#棋盘格模板规格
w = 9
h = 6
pattern_size=(w,h)
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
img_left_points=[]
img_right_points=[]
obj_points=[]
pattern_points       = np.zeros( (w*h, 3), np.float32 )
pattern_points[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

images_left = glob.glob('Left/*.png')
images_right = glob.glob('Right/*.png')

imgg=cv2.imread(images_right[0],cv2.CV_8UC1)
#image_size=imgg.shape
image_size=(332,252)
print "image_size:",image_size
cv2.imshow("imgg",imgg)

for i in range(len(images_left )):
#for i in range(9):
    #left_img = cv2.imread(images_left[i], cv2.CV_8UC1)
    #right_img = cv2.imread(images_right[i], cv2.CV_8UC1)
    left_img = cv2.imread(images_left[i])
    left_img=cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
    right_img = cv2.imread(images_right[i])
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("left", left_img)
    find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    left_found, left_corners = cv2.findChessboardCorners(left_img, pattern_size, flags=find_chessboard_flags)
    right_found, right_corners = cv2.findChessboardCorners(right_img, pattern_size, flags=find_chessboard_flags)

    if left_found:
        cv2.cornerSubPix(left_img, left_corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    if right_found:
        cv2.cornerSubPix(right_img, right_corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    if left_found and right_found:
        img_left_points.append(left_corners)
        img_right_points.append(right_corners)
        obj_points.append(pattern_points)

    cv2.imshow("left", left_img)
    cv2.drawChessboardCorners(left_img, pattern_size, left_corners, left_found)
    cv2.drawChessboardCorners(right_img, pattern_size, right_corners, right_found)

    cv2.imshow("left chess", left_img)
    cv2.imshow("right chess", right_img)
cv2.destroyAllWindows()
stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj_points,   img_left_points,  img_right_points, image_size, criteria=stereocalib_criteria, flags=stereocalib_flags)
print cameraMatrix1
print distCoeffs1
print cameraMatrix1
print distCoeffs2
print R
print T

