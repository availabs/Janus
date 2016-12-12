import numpy as np
from math import sqrt
from functools import reduce
# This class will be for keeping track of face boundingboxes
# It will leverage spacial as information along with descriptors
# To make descisions about if faces are new/existing 
class FaceDictionary:
    def __init__(self,bboxes=[],descriptors=[]):
        # create a dictionary of the bounding boxes with int ids
        self.bboxes = {k:v for k,v in enumerate(bboxes)}
        # create a dictionary of the descriptors with the same ids
        self.descriptors = {k:v for k,v in enumerate(descriptors)}
        # note the next available index, will be used when merging
        self.index = len(bboxes)
        # guarantee that there are the same number of boxes and descriptors
        assert (len(self.bboxes) == len(self.descriptors)), \
            "Must have the same number of boxes as descriptors"

    def getobjs(self):
        indexes = self.bboxes.keys()
        boxes = self.bboxes.values()
        descr = self.descriptors.values()
        for i,(b,(i1,b1)) in zip(indexes,(zip(boxes,self.bboxes.items()))):
            assert i == i1, "indexes don't match"
            assert b == b1, "boxes don't match"
        for i,(d,(i1,d1)) in zip(indexes,zip(descr,self.descriptors.items())):
            assert i == i1, "indexes don't match"
            assert d == d1, "boxes don't match"
        return indexes,boxes,descr
        
    # Method to update the dictionary with currenttime box & descriptors
    def update(self,boxes,descriptors):
        # get the length of the new items
        bl = len(boxes)
        dl = len(descriptors)
        # assert they are 1-1
        assert bl == dl,"Must have the same number of boxes and descriptors"
        
        oldboxes = self.bboxes
        olddesc  = self.descriptors
        # get time merged objects
        boxes,desc = self.merge(boxes,descriptors,oldboxes,olddesc)
        self.boxes = boxes
        self.descriptors = desc

        
    # method to merge the old boxes/descs with the new ones
    def merge(self,boxes,descs,oldbox,olddesc):
        boxcomps = {}
        descomps = {}
        newboxs = {}
        newdesc = {}
        # gen objs with comp scores between new box and old
        for key,box in enumerate(boxes):
            boxcomps[key] = {k:bcompare(box,v) for k,v in oldbox.items()}
        # gen objs with comp scores between new desc and old
        for key,desc in enumerate(descs):
            descomps[key] = {k:dcompare(desc,v) for k,v in olddesc.items()}

        #use comparisons to make descision about which box is assigned
        # an existing or new key and which are discarded
        for key,(box,desc) in enumerate(zip(boxes,descs)):
            # get the key with minimum spacial distance to last box
            (bk,bv) = reduce(lambda kv1,kv2: kv1 if kv1[1] <= kv2[1] else kv2,boxcomps[key])
            # get the key with minimum discription disparity to last one
            (dk,dv) = reduce(lambda kv1,kv2: kv1 if kv1[2] > kv2[1] else kv2, descomps[key])
            # if the box and discriptor key are the same ~~~~assume?success~~~~~ and from previous
            if dk == bk:
                newboxs[bk] = box
                newdesc[dk] = desc
            # if the face is close by the descriptor score
            # set it to the descriptors index
            elif dv > 0.8:  
                 newboxs[dk] = box
                 newdesc[dk] = desc
            # if the face is spacially within the hypotenuse of the
            # current box set it to the box's index !!assumes low motion
            elif bv < hypot(box):
                newboxs[bk] = box
                newdesc[bk] = desc
            # Otherwise label it as a new object
            else:
                newboxs[self.index] = box
                newdesc[self.index] = desc
                self.index += 1
            
# calculate the hypotenuse length of a rectangle
def hypot(box):
    return sqrt((box.right()-box.left())**2 + (box.bottom()-box.top())**2)
    
    #comparison score between two bboxes
    # will use squared distance between centroids
def bcompare(box1,box2):
    x1 = (box1.left() + box1.right())/2.0
    y1 = (box1.top() + box1.bottom())/2.0
    x2 = (box2.left() + box2.right())/2.0
    y2 = (box2.top() + box2.bottom())/2.0

    return (x2 - x1)**2 + (y2 - y1)**2
    
#comparison score between two descriptors
# try using cosine distance
def dcompare(desc1,desc2):
    return np.dot(desc1.normalize(),desc2.normalize())
