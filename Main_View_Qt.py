from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import * 
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import * 



import threading    as THREAD
import time         as TIME
import os           as OS
import sys          as SYSTEM
import datetime     as DATETIME

import head_rotation_detect as HEAD
from HistoryDict import *

path = OS.getcwd()
qtCreatorFile = path + OS.sep + "ui" + OS.sep + "mainView.ui"  
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)   

class Worker(QtCore.QThread):
    message = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.working = True

    def __del__(self):
        self.working = False
        self.wait()

    def run(self):
        while self.working == True:
            self.message.emit()
            TIME.sleep(0.08)


class MainUi(QtWidgets.QMainWindow, Ui_MainWindow):
            


    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        
        self.label_imageView        = self.findChild(QLabel, name='imageView')   
        self.label_fpsVal           = self.findChild(QLabel, name='labelShow_FPS_Rate')         
        self.label_horizonVal       = self.findChild(QLabel, name='labelShow_Angle_Horizon')   
        self.label_verticalVal      = self.findChild(QLabel, name='labelShow_Angle_Vertical')  
        self.label_headDirection    = self.findChild(QLabel, name='labelShow_Head_Direction')  

        self.label_horizonThreshold = self.findChild(QLabel, name='labelText_Angle_Threshold_Horizon') 
        self.label_verticalThreshold= self.findChild(QLabel, name='labelText_Angle_Threshold_Vertical')
        self.scrollBar_horizon      = self.findChild(QSlider, name='sliderBar_Threshold_Hor
        self.scrollBar_vertical     = self.findChild(QSlider, name='sliderBar_Threshold_Vertical')   
        self.label_focusRate        = self.findChild(QLabel, name='labelShow_Focus_Rate')        

        self.button_reset           = self.findChild(QPushButton, name='pushButton_Reset')
        self.checkBox_show3D        = self.findChild(QCheckBox, name='checkBox_Show_3D')
        self.checkBox_showFaceMark  = self.findChild(QCheckBox, name='checkBox_Show_Facemark')
        
        self.initVideoEvent()
        
        
        self.update_flag = True     
        self.fps_rate = 0          
        self.pre_time = TIME.time()

        self.angle_history = History( 2, max_size=20 )  
        self.focus_count        = 0
        self.focus_timer        = 0


        def reset():
            self.focus_count        = 0
            self.focus_timer        = 0
        self.button_reset.clicked.connect( reset )


        self.thread = Worker()
        self.thread.message.connect(self.update)
        self.thread.start()

    def initVideoEvent(self):
        pixmap = QPixmap('image.jpg')
        self.label_imageView.setPixmap(pixmap)


    def updateUI_second(self):
        pass
            
    
    def update(self):
        qApp.processEvents()


        show3D      = self.checkBox_show3D.isChecked()
        showMark    = self.checkBox_showFaceMark.isChecked()

        (horizon_angle, vertical_angle, cvimg) = HEAD.detect( show3D, showMark )

        try:

            image = QtGui.QImage(cvimg.data, cvimg.shape[1], cvimg.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            image = QtGui.QPixmap.fromImage( image )
            self.label_imageView.setPixmap(image)
        except:
            pass

        hThreshold = self.scrollBar_horizon.value()
        vThreshold = self.scrollBar_vertical.value()
        self.label_horizonThreshold.setText('水平閥值：     ' + str(hThreshold) )
        self.label_verticalThreshold.setText('垂直閥值：     ' + str(vThreshold) )


        self.angle_history.add( [horizon_angle, vertical_angle] )
        if( self.angle_history.is_full() ):
            [horizon_angle, vertical_angle] = self.angle_history.get_average()
            self.angle_history.pop()

        self.label_horizonVal.setText(  str( horizon_angle ) )
        self.label_verticalVal.setText( str( vertical_angle ) )

        (head_rotation, head_rotation_str) = HEAD.judge_look(horizon_angle       =horizon_angle, vertical_angle    =vertical_angle, 
                                                                horizon_threshold   =hThreshold   , vertical_threshold=vThreshold)
        
        rotation_str = '目前頭部方向： ' + str(head_rotation_str)
        self.label_headDirection.setText( rotation_str )

        if head_rotation == HEAD.LEFT or head_rotation == HEAD.RIGHT or head_rotation == HEAD.GONE:
            self.focus_count += 1
        self.focus_timer += 1
        self.label_focusRate.setText( str( round( (1 - (self.focus_count / self.focus_timer)) * 100, 2)  ) + '%' )
        
        
     
        self.fps_rate += 1
        now_time = TIME.time()
        if (now_time - self.pre_time) > 1:
            self.updateUI_second()

            self.label_fpsVal.setText( str( self.fps_rate ) )
            self.fps_rate = 0
            self.pre_time = now_time
    


            




if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication( SYSTEM.argv )
        window = MainUi()
        window.show()
        app.exec_()

    run_app()