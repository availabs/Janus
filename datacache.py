from facedetection.detection import myfacefinder as finder
from acquire import faceprocessor
from featureExtractor import modelInferance
from datetime import datetime
import numpy as np
import os
import cv2
import argparse


class feature_cacher:
    def __init__(self,finder=finder,
                 processor=faceprocessor,
                 extractor=modelInferance,
                 cachePath="/home/avail/data/facerecognition/cache"):
        
        self.processor = processor()
        self.processor.setFinder(finder().nextFaces)
        self.extractor = extractor()
        self.saveDir = cachePath

    def genFeaturePairs(self):
        temp = self.processor.getFaces()
        if temp is None:
            return None
        frame,f,wfaces = temp
        feats = self.extractor.getFeatures(wfaces)
        return zip(feats,wfaces)
            
    def cachePairs(self,pairs,label):
        date = lambda : '_'.join(datetime.today().__str__().split())
        print(date())
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        for feat,wface in pairs:
            filelabel = label + '_' + date()
            cv2.imshow('window',wface)
            #save the feature
            np.savetxt(os.path.join(self.saveDir,filelabel+'.npy'),feat)
            #save the face
            cv2.imwrite(os.path.join(self.saveDir,filelabel+'.jpg'),wface)

    def test(self,label='1'):
        print('made it')
        cv2.namedWindow('window')
        key = cv2.waitKey(10)
        while key < 0:
            print(key)
            key = cv2.waitKey(10)
            featpairs = self.genFeaturePairs()
            if featpairs is not None:
            #save features and pairs
                print('caching')
                self.cachePairs(featpairs,label)

    def cacheprocess(self,label='1',messenger=None):
        print('made it')
        cv2.namedWindow('window')
        key = cv2.waitKey(10)
        if messenger is None:
            print('No process messenger,exiting')
            return
        print (self.processor.finder)
        while key < 0 and messenger.poll(0.001) is False:
            print(key,messenger.poll(0.001))
            key = cv2.waitKey(10)
            featpairs = self.genFeaturePairs()
            if featpairs is not None:
                print('caching')
                self.cachePairs(featpairs,label)
        cv2.destroyAllWindows()
        print("Exiting Cacher with code {}".format(messenger.recv()) )
        
            
def getargs():
    ap = argparse.ArgumentParser()
    ap.add_argument('label',help='Must include the label to give the current subject')
    return ap.parse_args()

                
def main():
    args = getargs()
    cacher = feature_cacher()
    cacher.test(label=args.label)


if __name__ == '__main__':
    main()
