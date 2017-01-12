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

    def isempty(self):
        return len(self.bboxes.keys()) == 0
        
    def getobjs(self):
        indexes = self.bboxes.keys()
        boxes = self.bboxes.values()
        descr = self.descriptors.values()
        # for i,(b,(i1,b1)) in zip(indexes,(zip(boxes,self.bboxes.items()))):
        #     assert i == i1, "indexes don't match"
        #     assert b == b1, "boxes don't match"
        # for i,(d,(i1,d1)) in zip(indexes,zip(descr,self.descriptors.items())):
        #     assert i == i1, "indexes don't match"
        #     assert np.array_equal(d,d1), "boxes don't match"
        return indexes,boxes,descr
        
    # Method to update the dictionary with currenttime box & descriptors
    def update(self,boxes,descriptors):
        # get the length of the new items
        bl = len(boxes)
        dl = len(descriptors)
        if self.isempty():
            self.__init__(boxes,descriptors)
        # assert they are 1-1
        assert bl == dl,"Must have the same number of boxes and descriptors"

        oldboxes = self.bboxes
        olddesc  = self.descriptors
        # get time merged objects
        boxes,desc = self.merge(boxes,descriptors,oldboxes,olddesc)
        self.bboxes = boxes
        self.descriptors = desc

        
    # method to merge the old boxes/descs with the new ones
    def merge(self,boxes,descs,oldbox,olddesc):
        boxcomps = {}
        
        descomps = {}
        olddesccomps = {}
        newboxs = {}
        newdesc = {}
        # gen objs with comp scores between new box and old
        for key,box in enumerate(boxes):
            #calculate the diagonal length of the box
            diag = hypot(box)
            boxcomps[key] = {}
            for k,v in oldbox.items():
                score = bcompare(box,v)
                if sqrt(score) <= 1.15*diag:
                    boxcomps[key][k] = bcompare(box,v)
            
        # gen objs with comp scores between new desc and old
        for key,desc in enumerate(descs):
            descomps[key] = {}
            for k,v in olddesc.items():
                score = dcompare(desc,v)
                if k in boxcomps[key]:
                    descomps[key][k] = score
                    olddesccomps[k] = {}
                    olddesccomps[k][key] = score
                    
        #initialize list to store key of merged items
        merged_items = []
        # Here will will go over the groups around old faces
        for k,group in olddesccomps.items():
            #get the descriptor and index of the one of maximum similarity
            
            (dk,dv) = reduce(lambda kv1,kv2: kv1 if kv1[1] > kv2[1] else kv2, group.items())
            #get the box metric score for that one as well
            bv = boxcomps[dk][k]
                        
            # if the face is close by the descriptor score
            # set it to the descriptors index
            if dv > 0.70:
                merged_items.append(dk)
                print ('dependent on descriptor',dv)
                newboxs[k] = boxes[dk]
                newdesc[k] = descs[dk]
            else:
                print('not similar enough')
                #print the score and the distance
                print (k,dk, "Cosine Dist: ",dv)
                print (k,dk, "Squared Centroid Dist: ", bv)
                if sqrt(bv) < 0.4*hypot(boxes[dk]):
                    print ('close enough to assume the same')
                    merged_items.append(dk)
                    newboxs[k] = boxes[dk]
                    newdesc[k] = descs[dk]

            #otherwise that face is no longer present do not add that key
            #to the updated list
        #For the items that were not added
        items_not_added = [key for key,b in enumerate(boxes) if key not in merged_items]
        for item in items_not_added:
            print("Adding Item")
            # get the newest available index
            ix = self.index
            # slot the item into memory
            newboxs[ix] = boxes[item]
            newdesc[ix] = descs[item]
            # update to next available index
            self.index += 1
            
        return newboxs,newdesc
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
    return np.dot(unit(desc1),unit(desc2))

#calculate the unit vector of the one passed
# --input numpy vector
def unit(vec):
    return vec/np.linalg.norm(vec)
