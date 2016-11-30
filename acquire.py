from facedetection.detection import AlignDlib as ad
from facedetection.tracking  import multitracker as tracker
import os
import sys
import cv2
import numpy as np
import time
#Create class to cache face images collected from
#the "interview" process
class faceprocessor:

    def __init__(self,facedim=96):
        self.currfaces = []
        self.frame=None
        self.facedim=facedim
        self.source =None
        self.finder=None
        self.tracker=None
        self.last = 0
    def setSource(self,source):
        self.source = source
    def setFinder(self,finder):
        self.finder = finder
    def setTracker(self,tracker):
        self.tracker = tracker
    def grabFrame(self):
        st,frame = self.source.read()
        if st:
            return frame
        else:
            return None
    def getFaces(self):
        bgrframe = self.grabFrame()
        if self.tracker is not None:
            if self.tracker.isempty():
                rgbframe,faces = self.finder(bgrframe)
                self.tracker.buildtracker(rgbframe,faces)
                print('initiating')
            elif time.clock() - self.last > 2:
                self.last = time.clock()
                rgbframe,faces = self.finder(bgrframe)
                faces = self.tracker.supplement(rgbframe,faces)
                print('supplementing')
            else:
                rgbframe = cv2.cvtColor(bgrframe,cv2.COLOR_BGR2RGB)
                self.tracker.trackobjs(rgbframe)
                facesobj = self.tracker.getupdates()
                faces = [face for lbl,face in facesobj]
        else:
            rgbframe,faces = self.finder(bgrframe)
        if faces is None:
            return
        if len(faces) == 0:
            return None
        resfaces = []
        for face in faces:
            resfaces.append(ad.align(imgDim=self.facedim,
                                     rgbImg=rgbframe,bb=face))
        return rgbframe,faces,resfaces

if __name__ == '__main__':
    print('unimplemented')
    
