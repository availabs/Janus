import os
from glob import glob
import numpy as np
import json

class datagenerator:
    def __init__(self,paths=['/home/avail/data/facerecognition/cache'
                             ,'/home/avail/data/facerecognition/persistance'],
                 types=['*.npy'],sorteddata=False,tr2te=0.7):
        self.paths = paths
        self.types = types
        self.features = None
        self.labels = None
        self.trindex = 0
        self.teindex = 0
        self.trainingfeats = None
        self.testingfeats  = None
        self.traininglbls = None
        self.testinglbls  = None
        self.sorteddata = sorteddata
        self.trfactor = tr2te
        self.currentcls = None
        self.classes = None
        self.classix = 0
        self.sectionedtestingfeats = None
        self.sectionedtestinglbls = None
    def queryfiles(self,paths=None,query=None):
        filenames = []
        types = self.types if query is None else query
        paths = self.paths if paths is None else paths
        patterns = [os.path.join(path,typ) for path in paths for typ in types]
        filenames = np.array([fname for pat in patterns for fname in glob(pat)])
        return filenames


    def loadfeatures(self):
        if self.features is None or self.labels is None:
            fnames = self.queryfiles()
            clsmap = json.loads(open('classmap.json').read())
            numkeys = len(list(clsmap.keys()))
            path2classix = lbl_vec(clsmap,numkeys)
        if self.features is None:
            self.features = np.array([np.loadtxt(fn) for fn in fnames])
            ixs = np.arange(0,len(self.features))
            mx = len(ixs)
            print (mx)
            tr = int(mx*self.trfactor)
            print(tr)
            self.trainingfeats = self.features[ixs[:tr]]
            self.testingfeats  = self.features[ixs[tr:]]
            
            
            self.labels = np.vstack(np.array(list(map(path2classix,fnames))))
            
            print('labels',self.labels)
            self.traininglbls = self.labels[ixs[:tr]]
            print(self.traininglbls)
            self.testinglbls  = self.labels[ixs[tr:]]

    def load_sectioned_features(self):
        if self.features is None or self.labels is None:
            fnames = self.queryfiles()
            clsmap = json.loads(open('classmap.json').read())
            numkeys = len(list(clsmap.keys()))
            mapper = lbl_vec(clsmap,numkeys)
        if self.features is None:
            self.features = np.array([np.loadtxt(fn) for fn in fnames])
            self.labels =   np.vstack(np.array(list(map(mapper,fnames))))
            clss = self.get_classes(fnames,clsmap)
            clsnums = list(set(clss))
            self.trainingfeats = {}
            self.traininglbls = {}
            self.testingfeats = {}
            self.testinglbls = {}
            for id in clsnums:
                ixs = clss == id
                feats = self.features[ixs]
                lbls   = self.labels[ixs]
                numf = len(feats)
                pivot = int(numf*self.trfactor)
                self.trainingfeats[id] = feats[:pivot]
                self.traininglbls[id]  = lbls[:pivot]
                self.testingfeats[id] = feats[pivot:]
                self.testinglbls[id]  = lbls[pivot:]
            self.classes = clsnums
    def get_classes(self,fnames,clsmap):
        return np.array(
            list(map(lambda x:clsmap[x[x.rfind('/')+1:x.find('_')]],
                     fnames)))
    
    def load_sorted_features(self):
        if self.features is None or self.labels is None:
            fnames = self.queryfiles()
            clsmap = json.loads(open('classmap.json').read())
            numkeys = len(list(clsmap.keys()))
            mapper = lbl_vec(clsmap,numkeys)
        if self.features is None:
            self.features = np.array([np.loadtxt(fn) for fn in fnames])
            self.labels =   np.vstack(np.array(list(map(mapper,fnames))))
            clss = self.get_classes(fnames,clsmap)
            clsnums = list(set(clss))
            srtdfeats = [self.features[clss==cIx] for  cIx in clsnums]
            srtdlabels= [self.labels[clss ==cIx] for cIx in clsnums]

            lengths = [int(len(x)*self.trfactor) for x in srtdlabels]

            self.trainingfeats = np.vstack([feats[:mx]
                                for feats,mx in zip(srtdfeats,lengths)])
            self.traininglbls  = np.vstack([lbls[:mx]
                                for lbls,mx in zip(srtdlabels,lengths)])
            self.testingfeats = np.vstack([feats[mx:]
                                for feats,mx in zip(srtdfeats,lengths)])
            self.testinglbls  = np.vstack([lbls[mx:]
                                for lbls,mx in zip(srtdlabels,lengths)])
            print(self.traininglbls,self.testinglbls)
            
    def next_train_batch(self,bsize=100):
        if not self.sorteddata:
            self.loadfeatures()
        else:
            self.load_sorted_features()
        maxi = len(self.trainingfeats)
        idx = np.arange(self.trindex,self.trindex+bsize)
        idx = np.mod(idx,maxi)
        self.trindex = (self.trindex + bsize) % maxi
        return self.trainingfeats[idx],self.traininglbls[idx]

    def updateclass(self):
        #if the current index is in range
        if self.classix < len(self.classes):
            #get the classid in that position
            self.currentcls = self.classes[self.classix]
            id = self.currentcls
            #if at the beginning set section to the first one
            if self.classix == 0:
                self.sectionedtestingfeats = self.testingfeats[id]
                self.sectionedtestinglbls  = self.testinglbls[id]
            #otherwise stack it ontop the previous class
            else:
                id = self.currentcls
                self.sectionedtestingfeats = np.vstack([
                    self.sectionedtestingfeats,self.testingfeats[id]])
                self.sectionedtestinglbls  = np.vstack([
                    self.sectionedtestinglbls,self.testinglbls[id]])
            #increment the class index for the next jump
            self.classix += 1
    
    def next_sectioned_train_batch(self,bsize=100,nextcls=False):
        self.load_sectioned_features()
        if self.currentcls is None:
            self.updateclass()
        if nextcls:
            self.updateclass()

        id = self.currentcls
        maxi = len(self.trainingfeats[id])
        idx = np.arange(self.trindex,self.trindex+bsize)
        idx = np.mod(idx,maxi)
        self.trindex = (self.trindex+bsize) % maxi
        return self.trainingfeats[id][idx],self.traininglbls[id][idx]
    
    def next_test_batch(self,bsize=100):
        if not self.sorteddata:
            self.loadfeatures()
        else:
            self.load_sorted_features()
        maxi = len(self.testingfeats)
        idx = np.arange(self.teindex,self.teindex+bsize)
        idx = np.mod(idx,maxi)
        self.teindex = (self.teindex + bsize) % maxi
        return self.testingfeats[idx],self.testinglbls[idx]


def lbl_vec(nmmap,length):

    def fun(name):
        name = name[name.rfind('/')+1:name.find('_')]
        cls = nmmap[name]
        lvec = np.zeros((1,length),dtype=np.uint8)
        lvec[0,cls] = 1
        return lvec
    return fun
    
if __name__ == '__main__':
    datagen = datagenerator(sorteddata=True)
    datagen.load_sorted_features()
        
