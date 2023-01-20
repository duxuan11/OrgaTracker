from fileinput import filename
from turtle import width
import cv2 as cv
import numpy as np
import os
def main():
    data_dir = "pix2pix/src5"
    img_list = os.listdir(data_dir)
    print(img_list)
    kernel = np.ones((2,2),np.uint8)
    kernel1 = np.ones((1,1),np.uint8)
    for img_file in img_list:
        if img_file.endswith(".png") or img_file.endswith(".jpg") or img_file.endswith("tif"):
            image=cv.imread(data_dir+"/"+img_file,cv.IMREAD_GRAYSCALE)
            gauss = cv.GaussianBlur(image,(5,5),25)
            gauss_canny = cv.Canny(gauss, 30, 70)
            gauss_canny = cv.erode(gauss_canny,kernel1)
            #edge = cv.Canny(image,50,100)
            height,width = 256,256
            
            gauss_canny = cv.dilate(gauss_canny,kernel)
            #cv.imshow("aa",gauss_canny)
            dst=np.zeros((height,width,1),np.uint8)    #将图片改为灰度图
            
            #黑白反转
            for i in range(0,height):
                for j in range(0,width):
                    grayPixel=gauss_canny[i][j]
                    dst[i,j]=255-grayPixel

            # cv.namedWindow('res', cv.WINDOW_NORMAL)

            # cv.moveWindow("res", 1000, 500) 
            # cv.imshow("res",dst)
            cv.imwrite("pix2pix/res/"+str(img_file.split('/')[-1]),dst)
            cv.waitKey(0)

def becomeblack():
    data_dir = "skeleton/src1"
    img_list = os.listdir(data_dir)
    height,width = 256,256
    for img_file in img_list:
        if img_file.endswith(".png") or img_file.endswith(".jpg") or img_file.endswith("tif"):
            image=cv.imread(data_dir+"/"+img_file,cv.IMREAD_GRAYSCALE)
            #edge = cv.Canny(image,50,100)
            dst=np.zeros((height,width,1),np.uint8)    #将图片改为灰度图
            print(image)
            #黑白反转
            for i in range(0,height):
                for j in range(0,width):
                    if image[i][j] < 200:
                        dst[i,j] = 0
                    else:
                        dst[i,j] = 255    

        
            cv.imwrite("skeleton/res2/"+str(img_file.split('/')[-1]),dst)           

if __name__=="__main__":
    main()