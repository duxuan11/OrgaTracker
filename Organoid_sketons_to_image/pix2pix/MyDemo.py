
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QLabel
from demo.Ui_MainWin import Ui_MainWindow
from PyQt5 import QtCore, QtGui
from demo.paintBoard import ShowBox,Drawing
from PyQt5.QtGui import QPixmap
from demo.Imagepro import generate
class MyWeight(QMainWindow,Ui_MainWindow): #这里也要记得改
    def __init__(self,parent =None):
        super(MyWeight,self).__init__(parent)
        self.setupUi(self)
        self.InputPaintBoard = Drawing()
        self.OutputPaintBoard = ShowBox()
        self.InputPaintBoard.setObjectName("InputPaintBoard")
        self.horizontalLayout_3.addWidget(self.InputPaintBoard)
        self.line = QtWidgets.QFrame(self.widget1)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_3.addWidget(self.line)
        self.OutputPaintBoard.setObjectName("OutputPaintBoard")
        self.horizontalLayout_3.addWidget(self.OutputPaintBoard)
        self.clearBtn.clicked.connect(self.clearCanvs)
        self.undoBtn.clicked.connect(self.undoCanvs)
        self.processBtn.clicked.connect(self.process)
        self.SaveBtn.clicked.connect(self.save)
        self.randomBtn.clicked.connect(self.imgrandom)
        self.undoBtn.clicked.connect(self.openfile)
    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.update()

    def clearCanvs(self):
        self.InputPaintBoard.clearCanvs()
        self.OutputPaintBoard.clearCanvs()
        #self.InputPaintBoard.update()
    def undoCanvs(self):
        self.InputPaintBoard.undo()

    def process(self):
        self.InputPaintBoard.save()  
        generate()
        self.imgshow()
        pass
    def save(self):
        pass  
    def imgshow(self):
        self.OutputPaintBoard.imgshow()
        pass
    def imgrandom(self):
        self.InputPaintBoard.imgrandom()

    def openfile(self):
        pass
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWin = MyWeight()
    myWin.show()
    sys.exit(app.exec_())   



