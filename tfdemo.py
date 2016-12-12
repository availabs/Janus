import pickle
import json
from facedetection.detection import myfacefinder as finder
from facedetection.tracking import multitracker as tracker
from acquire import faceprocessor
from featureExtractor import modelInferance
from datacache import feature_cacher
from facedictionary import FaceDictionary
import numpy as np
import cv2
import svmclassifier
from multiprocessing import Process, Lock, Pipe, Queue
import time
from tensorflowclassifier import model
from datagen import datagenerator
from IPython import embed
from windowmaintainer import window
from tts import mptts
import queue



def mainloop(pipe=None):
    classifierLock = Lock()
    p1_conn, ch1_conn = Pipe()
    p2_conn, ch2_conn = Pipe()
    tq = Queue(1)
    main = Process(target=outfacing, args=(classifierLock,ch1_conn))#,tq))
    #tts =  Process(target=mptts, args=(tq,))
    main.start()
    #tts.start()
    print('made it')
    lastback = time.time()
    while True:
        m = main.exitcode
        
        
        if m is not None:
            break;
        
        if pipe and pipe.poll(0.0001):
            signal = pipe.recv()
            print(signal)
            if signal == 'STOP':
                p1_conn.send('KILL')
                break
            elif signal.startswith('label:') or signal == 'STOPCACHE':
                p1_conn.send(signal)

    #Send the kill signal to tts server
    #tq.put('KILL')
    #Wait for the process to term
    #tts.join()
    
    
def load_classifier(lock):
    lock.acquire()
    clf = None
    try:
        clf = pickle.load(open('currbestsvm.pkl','rb'))
    finally:
        lock.release()
    classmap = json.loads(open('classmap.json').read())
    revmap = {v:k for k, v in classmap.items()}


    return clf,revmap
def outfacing(lock,receiver,ttsque=None,finder=finder):
    cv2.namedWindow('visuals')
    processor = faceprocessor()
    extractor = modelInferance()
    processor.setSource(cv2.VideoCapture(0))
    processor.setFinder(finder().nextFaces)
    faceDict = FaceDictionary()
    datawindow = window()
    dnn = model()
    dnn.load_previous()
    generator = datagenerator()
    clf,revmap = load_classifier(lock)
    cacher = feature_cacher()
    kill = False
    cacheFlag = False
    label = None
    cv2.namedWindow('visual')
    ix = 0
    while cv2.waitKey(10) < 0 and not kill:
        temp = processor.getFaces()
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
        if temp is None:
            continue
        frame,fbbs,wfaces = temp
        feats = extractor.getFeatures(wfaces)
        #faceDict.update(fbbs,feats)
        #index,fbbs,feats = faceDict.getobjs()
        index = np.ones(len(fbbs))
        clss,score,tmap = handledata(dnn,cacher,generator,cacheFlag,label,
                                       feats,wfaces,datawindow)
        if tmap is not None:
            revmap = tmap
        if score is not None:
            isUnknown = score < 8.0
        else:
            isUnknown = False

        sendmesg(isUnknown,'What are you doing here? Relinquish thine name. ',ttsque)

        ix += 1
        
        visualize(frame,fbbs,wfaces,clss,revmap,isUnknown,index)
        # print(clss)
    extractor.kill()
    return 1


def sendmesg(flag,msg,ttsque):
    if flag and ttsque is not None:
        try:
            ttsque.put(msg,False)
        except queue.Full:
            print('q b full')
    
def visualize(frame,fbbs,wfaces,clss,revmap,unknown,index):
    pair = wfaces[0].shape
    vizframe = np.zeros((pair[0],pair[1]*len(wfaces),3),dtype=np.uint8)

    #clss = clf.predict_proba(feats)
    i = 0
    for face,bbx,probs,ix in zip(wfaces,fbbs,clss,index):
        #retrieve the top 2 classes based on probabilty
        top2 = probs.argsort()[-2:] 
        for ix,cls in enumerate(top2):
            cv2.putText(face,revmap[round(cls)]+':'+str(probs[cls]),
                        (0,45+ix*50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
            vizframe[:,i*96:(i+1)*96] = face

        i += 1
        topcorner = (bbx.left(),bbx.top())
        textcorner = (bbx.left(),bbx.top()-15)
        idcorner   = (bbx.right()-10,bbx.top())
        botcorner = (bbx.right(),bbx.bottom())
        cv2.rectangle(frame,topcorner,botcorner,(0,255,0))
        cv2.putText(frame,revmap[round(top2[1])]+':{0:.3f}'.format(probs[top2[1]]),
                    textcorner,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
        cv2.putText(frame,'id:{}'.format(ix),idcorner,
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))

        if(unknown):
            cv2.putText(frame,'Unknown Exists',(25,425),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255))
    cv2.imshow('visual',frame)
    cv2.imshow('visuals',vizframe)


def handledata(dnn,cacher,generator,cacheFlag,label,feats,wfaces,datawindow):
    preds = None
    score  = None
    revmap = None
    if cacheFlag:
        pairs = zip(feats,wfaces)
        trfeats,trlbls,newclsmap = generator.livefeed(np.vstack(feats),
                                            np.repeat(label,len(feats)))
        
        revmap = {v:k for k,v in newclsmap.items()}
        print('Attempting Online learning')
        #cacher.cachePairs(pairs,label)
        dnn.input_traindata(trfeats,trlbls)
        dnn.backprop()
        dnn.input_data(np.vstack(feats))
        preds = dnn.pred()
    else:
        dnn.input_data(np.vstack(feats))
        preds = dnn.pred()
        #this is assuming ONLY ONE person at a time 
        datawindow.push(preds[0])
        if datawindow.isfull():
            ps = datawindow.getobjs()
            ps = np.vstack(list(ps))
            scores = dnn.non_class_window_score(ps,datawindow.ws)
            score = scores.mean()
            
        
    
    
    return preds,score,revmap


if __name__ == '__main__':
    mainloop()
