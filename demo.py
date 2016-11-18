import pickle
import json
from facedetection.detection import myfacefinder as finder
from acquire import faceprocessor
from featureExtractor import modelInferance
import numpy as np
import cv2
import svmclassifier
from multiprocessing import Process, Lock, Pipe
import time

def main():
    classifierLock = Lock()
    p1_conn, ch1_conn = Pipe()
    p2_conn, ch2_conn = Pipe()
    main = Process(target=outfacing, args=(classifierLock,ch1_conn))
    back = Process(target=svmclassifier.train, args=(classifierLock,p1_conn))
    main.start()
    back.start()
    lastback = time.time()
    while True:
        b = back.exitcode
        m = main.exitcode
   
        if m is not None:
            if b is None:
                back.join()
            break;
        if b is not None and time.time() - lastback > 60:
            back = Process(target=svmclassifier.train, args=(classifierLock,p1_conn))
            back.start()
            lastback = time.time()
    
def load_classifier(lock):
    lock.acquire()
    clf = None
    try:
        clf = pickle.load(open('currbestsvm.pkl','rb'))
    finally:
        lock.release()
    classmap = json.loads(open('classmap.json').read())
    revmap = {}
    for key,val in classmap.items():
        revmap[val] = key

    return clf,revmap
def outfacing(lock,receiver,finderfun=finder().nextFaces):
    cv2.namedWindow('visuals')
    processor = faceprocessor()
    extractor = modelInferance()
    processor.setFinder(finderfun)
    clf,revmap = load_classifier(lock)
    
    while cv2.waitKey(10) < 0:
        temp = processor.getFaces()
        if temp is None:
            continue
        frame,f,wfaces = temp
        feats = extractor.getFeatures(wfaces)
        pair = wfaces[0].shape
        vizframe = np.zeros((pair[0],pair[1]*len(wfaces),3),dtype=np.uint8)
        if receiver and receiver.poll(0.001):
            print(receiver.recv())
            clf,revmap = load_classifier(lock)
            print ('HOT SWAPPED CLASSIFIER')
        clss = clf.predict(feats)
        i = 0
        for face,cls in zip(wfaces,clss):
            cv2.putText(face,revmap[cls],(0,95),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            vizframe[:,i*96:(i+1)*96] = face
            i += 1
        cv2.imshow('visuals',vizframe)
    return 1

if __name__ == '__main__':
    main()
