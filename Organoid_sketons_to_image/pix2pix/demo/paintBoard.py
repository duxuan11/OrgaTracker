from typing_extensions import Self
from PyQt5 import QtGui, QtCore, QtWidgets 
from sklearn.metrics import pairwise 
import numpy as np
import math
from PyQt5.QtCore import Qt,QSize,QPoint
from PyQt5.QtGui import QPainter, QPixmap,QPalette,QPen,QColor
import sys 
import os

class Drawing(QtWidgets.QWidget):
    def __init__(self):
        super(Drawing, self).__init__()
        self.setWindowTitle("绘图应用")
        self.pix = QPixmap()
        # 画布大小为400*400, 背景为白色
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        
        self.initUI()
        self.img_list = os.listdir('demo/Img/display')
        self.count = 0
        self.max = len(self.img_list)
 
    def initUI(self):  
        self.resize(260, 260)
        #画布大小为400*400, 背景为白色
        self.pix = QPixmap(256, 256)
        self.pix.fill(Qt.white)

    #进行绘制操作
    def paintEvent(self, event):
        pp = QPainter(self.pix)
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        pp.setPen(pen)
        #根据鼠标指针前后两个位置绘制直线
        pp.drawLine(self.lastPoint, self.endPoint)
        #让前一个坐标值等于后一个坐标值
        self.lastPoint = self.endPoint
        #在画板上进行绘制操作
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)

    #当鼠标被第一次点击时, 记录开始的位置
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()

    #当鼠标移动时，开始改变最后的位置，且进行绘制操作
    def mouseMoveEvent(self, event):
        if event.button and Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        #鼠标左键释放,在进行最后一次绘制
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            #进行重新绘制
            self.update()


    def clearCanvs(self):
        self.pix.fill(Qt.white)
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.update()
    def save(self):
        self.pix.save("demo/res.jpg")    

    def imgrandom(self):
        filename = self.img_list[self.count]
        self.pix.load(os.path.join("demo/Img/display/",filename))
        self.update()
        self.count = self.count+1
        if self.count == self.max:
            self.count = 0
        print(self.count)    
        pass

class ShowBox(QtWidgets.QWidget):
    def __init__(self):
        super(ShowBox, self).__init__()
        self.setWindowTitle("绘图应用")
        self.pix = QPixmap()
        # 画布大小为400*400, 背景为白色
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.initUI()

    def initUI(self):
        self.setStyleSheet('border-width: 1px;border-style: solid;border-color:black ')    
        self.resize(256, 256)
        #画布大小为400*400, 背景为白色
        self.pix = QPixmap(256, 256)
        self.pix.fill(Qt.white)
    
    #注意qpixmap想要生效必须要有qpainter绘画操作
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pix)

    def imgshow(self):
        self.pix.load("results/edges2inst_pix2pix/test_latest/images/example_fake_B.png")
        self.update() 

    def clearCanvs(self):
        self.pix.fill(Qt.white)
        self.update()    
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget    
   
def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = Drawing()
    gui.show()
    sys.exit(app.exec_())
 
if __name__ == '__main__':
    main()
