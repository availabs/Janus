import cv2
import os
import dlib
from openface import AlignDlib as AD
import numpy as np
import time

AlignDlib = AD('/home/avail/code/torch-projs/openface/models/dlib/shape_predictor_68_face_landmarks.dat')


class myfacefinder:
    
    def __init__(self,scale=0.75,verb=False):
        self.face_finder = self.build_facefinder()
        self.scale = scale
        self.verb = verb
    def build_facefinder(self):
        return AlignDlib.getAllFaceBoundingBoxes

    def nextFaces(self,inframe=None):
        if inframe is None:
            return (inframe,None)
        frame = inframe
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        f = cv2.resize(rgb,(0,0),
                       fx=self.scale,fy=self.scale)
        dfaces = self.face_finder(f)
        patches = []
        truefaces = []
        for box in dfaces:
            sc = 1/self.scale
            top=(round(sc*box.left()),round(sc*box.top()))
            bot=(round(sc*box.right()),round(sc*box.bottom()))
            x1,y1 = top
            x2,y2 = bot
            truefaces.append(dlib.rectangle(left=x1,top=y1,right=x2,bottom=y2))
            if self.verb:
                patches.append(frame[y1:y2,x1:x2])
        return rgb,truefaces

def main():
    import pdb
    pdb.set_trace()
    finder = myfacefinder(verb=True)
    while True:
        faces = finder.nextFaces()
        if faces is None:
            break
        print(len(faces))
    
if __name__ == '__main__':
    main()
