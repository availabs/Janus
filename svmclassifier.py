import numpy as np
from sklearn import svm
import os
from glob import glob
import time
import pickle
import json
tol = 0.34


def load_names(paths):
        fnames = [name for path in paths for name in glob(os.path.join(path,'*_*.npy'))]
        print ( len(fnames),np.array(fnames).shape)
        clsnames = map(lambda x: x[x.rfind('/')+1:x.find('_')],
                       fnames)
        clsnames = list(set(clsnames))
        print(clsnames,paths)
        return clsnames,fnames
def un_cache(paths):
        for path in paths:
                if path.find('cache') >=0:
                        os.rename(path,path.replace('cache','persistance'))
def try_action(fun,args={},datalock=None):
    #grab the data lock if necessary
    if datalock is not None: datalock.acquire()
    try:
        retval = fun(**args)
    finally:
        if datalock is not None: datalock.release()
    return retval
    
def write_classmap(classmap):
    with open('classmap.json','w') as f:
        f.write(json.dumps(classmap))
    
def train(lock=None,sender=None):
    dirpath = '/home/avail/data/facerecognition/cache'
    permpath= '/home/avail/data/facerecognition/persistance'
    classes = {}
    clsnames,fnames =try_action(load_names,{'paths':[dirpath,permpath]})
    classmap = {}
    maxID = 0
    for lbl,cls in enumerate(clsnames):
        classes[cls] = {}
        classes[cls]['feats'] = np.array([np.loadtxt(x)
                                          for x in filter(lambda st: st.find(cls) > 0,fnames)])
        classes[cls]['len'] = classes[cls]['feats'].shape[0]
        print(classes[cls]['len'])
        classes[cls]['lbl'] = lbl*np.ones((classes[cls]['len'],),dtype=np.uint8)
        classmap[cls] = lbl
        maxID = lbl
    write_classmap(classmap)




    accs = []
    classifiers = []
    for valtrainratio in np.linspace(0.1,0.8,num=3):
        Xtr = np.array([])
        Ytr = np.array([])
        Xval = np.array([])
        Yval = np.array([])
        for i,cls in enumerate(classes):
            clslen = classes[cls]['len']
            feats = classes[cls]['feats']
            lbls = classes[cls]['lbl']
            vali = np.random.choice(2,clslen,
                                    p=[1-valtrainratio,valtrainratio]).astype(np.bool)
            if i == 0:
                Xtr = feats[~vali]
                Ytr = lbls[~vali]
                Xval = feats[vali]
                Yval = lbls[vali]
                
            else:
                Xtr = np.concatenate((feats[~vali],Xtr)) 
                Ytr = np.concatenate((lbls[~vali],Ytr)) 
                
                Xval = np.concatenate((feats[vali],Xval)) 
                Yval = np.concatenate((lbls[vali],Yval))
            print (Xtr.shape)
    
        start = time.clock()
        
        clf = svm.SVC(kernel='linear',C=100,probability=True)
        clf.fit(Xtr,Ytr)
        print ('Using {} samples,training time: {}'.format(Ytr.shape[0],time.clock()-start))
        preds = clf.predict(Xval)
        print(clf.predict_proba(Xval))
        for i in range(0,maxID+1):
                preds = np.round(preds)
                
        acc = np.sum(preds == Yval)/Yval.shape[0]
        accs.append(acc)
        classifiers.append(clf)
        print ('linear',100,'Accuracy : {}'.format(acc))

    best_model_index = np.array(accs).argmax(axis=0)
    print('best accuracy: {}'.format(accs[best_model_index]))
    bestclf = classifiers[best_model_index]
    try_action(write_model,{'model':bestclf},datalock=lock)
    notify(sender)
    un_cache(fnames)
    return 1
def write_model(model):
        with open('currbestsvm.pkl','wb') as output:
                pickle.dump(model,output,protocol=2)

def notify(sender):
        if sender is not None:
                sender.send('Done Thinking')
if __name__ == '__main__':
    train()
