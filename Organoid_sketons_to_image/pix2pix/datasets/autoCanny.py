import cv2 as CV2
import numpy as np
import os
root = os.getcwd()
# img = CV2.imread(root+"\\organoid_out\\00005.jpg", 0)
img = CV2.imread(root+"\\leinao.jpg", 0)
#高斯滤波
gauss = CV2.GaussianBlur(img,(5,5),0,0)
gauss_canny = CV2.Canny(gauss, 10, 120)
CV2.imshow("gauss_canny", gauss_canny)
CV2.imwrite(root+"\\2.jpg",gauss_canny)

#计算梯度值
gradx = CV2.Sobel(img,CV2.CV_64F,1,0)
grady = CV2.Sobel(img,CV2.CV_64F,0,1)
value = np.abs(gradx) + np.abs(grady)
angle = np.arctan(grady/(gradx+0.001))
#非极大值抑制
h,w = value.shape
img_thin = np.zeros_like(value)
for i in range(1, h-1):
    for j in range(1, w-1):
        theta = angle[i,j]
        if -np.pi / 8 <= theta < np.pi / 8:
            if value[i,j] == max(value[i,j],value[i,j-1],value[i,j+1]):
                img_thin[i,j] = value[i,j]
        elif -3*np.pi / 8 <= theta < -np.pi / 8:
            if value[i,j] == max(value[i,j],value[i,j-1],value[i,j+1]):
                img_thin[i,j] = value[i,j]
        elif np.pi / 8 <= theta < 3*np.pi / 8:
            if value[i,j] == max(value[i,j],value[i,j-1],value[i,j+1]):
                img_thin[i,j] = value[i,j]
        else:
            if value[i,j] == max(value[i,j],value[i,j-1],value[i,j+1]):
                img_thin[i,j] = value[i,j]
#基于梯度强度统计信息计算法实现自适应阈值
MAX = gauss.max()
MIN = gauss.min()
MED = np.median(gauss)
average = (MAX + MIN + MED) / 3       #最大值，最小值，中位数求平均
sigma = 0.33                          #sigma用于更改基于梯度强度统计信息确定阈值的百分比
th1 = max(0, (1-sigma)* average)      #低阈值
th2 = min(255, (1+sigma)* average)    #高阈值



#双阈值检测
h,w = img_thin.shape
img_edge = np.zeros_like(img_thin,dtype=np.uint8)
for i in range(1, h-1):
    for j in range(1, w-1):
        if img_thin[i,j] >= th2:
            img_edge[i,j] = 255
        elif img_thin[i,j] > th1:
            around = img_thin[i-1 : i+2, j-1 : j+2]
            if around.max() >= th2:
                img_edge[i,j] = 255

CV2.imshow("img_edge",img_edge)
CV2.imwrite("D:\img_edge.jpg",img_edge)
CV2.waitKey()
CV2.destroyAllWindows() 