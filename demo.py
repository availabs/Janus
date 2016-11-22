import pickle
import json
from facedetection.detection import myfacefinder as finder
from acquire import faceprocessor
from featureExtractor import modelInferance
from datacache import feature_cacher
import numpy as np
import cv2
import svmclassifier
from multiprocessing import Process, Lock, Pipe
import time

def mainloop(pipe=None):
    classifierLock = Lock()
    p1_conn, ch1_conn = Pipe()
    p2_conn, ch2_conn = Pipe()
    main = Process(target=outfacing, args=(classifierLock,ch1_conn))
    back = Process(target=svmclassifier.train, args=(classifierLock,ch2_conn))
    main.start()
    back.start()
    lastback = time.time()
    while True:
        b = back.exitcode
        m = main.exitcode

        if p2_conn.poll(0.0001):
            signal = p2_conn.recv()
            if signal == 'Done Thinking':
                pipe.send(signal)
            
            p1_conn.send(signal)
        
        if m is not None:
            if b is None:
                back.join()
            break;
        if pipe is None and b is not None and time.time() - lastback > 60:
            back = Process(target=svmclassifier.train, args=(classifierLock,ch2_conn))
            back.start()
            lastback = time.time()
        if pipe and pipe.poll(0.0001):
            signal = pipe.recv()
            print(signal)
            if signal == 'STOP':
                if b is None:
                    back.join()
                p1_conn.send('KILL')
                break
            elif signal.startswith('label:') or signal == 'STOPCACHE':
                p1_conn.send(signal)
            elif signal.startswith('LEARN'):
                if b is not None:
                    back = Process(target=svmclassifier.train, args=(classifierLock,ch2_conn))
                    back.start()
                
            
    
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
def outfacing(lock,receiver,finder=finder):
    cv2.namedWindow('visuals')
    processor = faceprocessor()
    extractor = modelInferance()
    processor.setFinder(finder().nextFaces)
    clf,revmap = load_classifier(lock)
    cacher = feature_cacher()
    kill = False
    cacheFlag = False
    label = None
    while cv2.waitKey(10) < 0 and not kill:
        temp = processor.getFaces()
        if temp is None:
            continue
        frame,f,wfaces = temp
        feats = extractor.getFeatures(wfaces)
        pair = wfaces[0].shape
        vizframe = np.zeros((pair[0],pair[1]*len(wfaces),3),dtype=np.uint8)
        if receiver and receiver.poll(0.001):
            signal = receiver.recv()
            print(signal)
            if signal.find('label:') >= 0:
                label = signal[signal.find(':')+1:]
                cacheFlag = True
            if signal == 'STOPCACHE':
                cacheFlag = False
            if signal == 'KILL':
                kill = True
            if signal == 'Done Thinking':
                clf,revmap = load_classifier(lock)
                print ('HOT SWAPPED CLASSIFIER')
        clss = clf.predict_proba(feats)
        if cacheFlag:
            pairs = zip(feats,wfaces)
            cacher.cachePairs(pairs,label)
        i = 0
        for face,probs in zip(wfaces,clss):
            #retrieve the top 2 classes based on probabilty
            top2 = probs.argsort()[-2:] 
            for ix,cls in enumerate(top2):
                cv2.putText(face,revmap[round(cls)]+':'+str(probs[cls]),
                            (0,45+ix*50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
                vizframe[:,i*96:(i+1)*96] = face

            i += 1
        cv2.imshow('visuals',vizframe)
    extractor.kill()
    return 1

if __name__ == '__main__':
    mainloop()
