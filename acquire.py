from facedetection.detection import AlignDlib as ad
import os
import sys
import cv2
import numpy as np

#Create class to cache face images collected from
#the "interview" process
class faceprocessor:

    def __init__(self,facedim=96):
        self.currfaces = []
        self.frame=None
        self.facedim=facedim

    def setFinder(self,finder):
        self.finder = finder
        print(finder)
        
    def grabFrame(self):
        print(self.finder)
        frame,faces = self.finder()
        if faces is not None:
            self.currfaces = faces
            self.frame=frame
            return True
        else:
            self.currfaces = None
            self.frame=frame
            return False

    def getFaces(self):
        self.grabFrame()
        if self.currfaces is None:
            return
        
        faces = self.currfaces
        frame = self.frame
        if len(faces) == 0:
            return None
        resfaces = []
        for face in faces:
            resfaces.append(ad.align(imgDim=self.facedim,
                                     rgbImg=frame,bb=face))
        return frame,faces,resfaces

if __name__ == '__main__':
    cacher = faceprocessor()
    cacher.test()
