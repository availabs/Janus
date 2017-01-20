import os
from glob import glob
import numpy as np
import json
import datetime
import pandas
def loadPandaFrame ( path ) :
    files = glob(os.path.join( path, '*_*' ) )
    trimer = lambda x : x.split('/')[-1]
    name = lambda x : x[:x.find('_')]
    date = lambda x : datetime.datetime(*tuple(map(int,x[x.find('_')+1:x.find('.npy')].replace('_','-').replace(':','-').replace('.','-').split('-'))))
    data = lambda x : np.loadtxt(x)
    loaded_data = map( lambda x : {'name':name(trimer(x)),
                                   'date':date(trimer(x)),
                                   'file':x},files)
    
    
    sorted(loaded_data,lambda (x,y): x.date - y.date)

            
    return df

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
        self.currentclss= []
        self.classes = None
        self.classix = 0
        self.classmap = None
        self.mixedindxs={}
        self.sectionedtestingfeats = None
        self.sectionedtestinglbls = None
        self.livestream = datagenstream()
        
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
            clsmap = json.loads(open('/home/avail/code/ROS/dl_ros_nodes/src/facerec/scripts/facerecognition/classmap.json').read())
            numkeys = len(list(clsmap.keys()))
            path2classix = mapper_lbl_vec(clsmap,numkeys)
        if self.features is None:
            self.features = np.array([np.loadtxt(fn) for fn in fnames])
            ixs = np.arange(0,len(self.features))
            mx = len(ixs)

            tr = int(mx*self.trfactor)

            self.trainingfeats = self.features[ixs[:tr]]
            self.testingfeats  = self.features[ixs[tr:]]
            
            
            self.labels = np.vstack(np.array(list(map(path2classix,fnames))))
            

            self.traininglbls = self.labels[ixs[:tr]]

            self.testinglbls  = self.labels[ixs[tr:]]

    def load_sectioned_features(self):
        if self.features is None or self.labels is None:
            #get filenames of the current files
            fnames = self.queryfiles()
            #get the classmap for those objects
            clsmap = json.loads(open('/home/avail/code/ROS/dl_ros_nodes/src/facerec/scripts/facerecognition/classmap.json').read())
            self.classmap = clsmap
            #get numerical ids for them
            numkeys = len(list(clsmap.keys()))
            #define a mapper from classes to keys
            mapper = mapper_name
        if self.features is None:
            #load the features
            self.features = np.array([np.loadtxt(fn) for fn in fnames])
            #load the labels
            self.labels =   np.vstack(np.array(list(map(mapper,fnames))))
            #get classes from the filenames
            clss = self.get_classes(fnames,clsmap)
            #get their id
            clsnums = list(set(clss))
            #initialize storage
            self.trainingfeats = {}
            self.traininglbls = {}
            self.testingfeats = {}
            self.testinglbls = {}
            # for each class
            for idx in clsnums:
                #add zero index to multiclass indexmap
                self.mixedindxs[idx] = 0
                #define numpy mask for the features of the desired class
                ixs = clss == idx
                #group them together
                feats = self.features[ixs]
                #as well as their labels
                lbls   = self.labels[ixs]
                numf = len(feats)
                #define a pivot between training and validation data
                pivot = int(numf*self.trfactor)
                #move the training and testing data to their class storage
                self.trainingfeats[idx] = feats[:pivot]
                self.traininglbls[idx]  = lbls[:pivot]
                self.testingfeats[idx] = feats[pivot:]
                self.testinglbls[idx]  = lbls[pivot:]
            self.classes = clsnums

    def get_classes(self,fnames,clsmap):
        return np.array(
            list(map(lambda x:clsmap[x[x.rfind('/')+1:x.find('_')]],
                     fnames)))

    #This function will handle the case of newly inputted data from 
    def livefeed(self,newfeats,labels=None):
        self.load_sectioned_features()
        if labels is not None:
            lbls = set(labels)
            exisitinglbls = set(self.classmap.keys())
            newlbls = lbls.difference(exisitinglbls)
            maxkey = max(self.classmap.values())
            for lbl in newlbls:
                maxkey += 1
                self.classmap[lbl] = maxkey
                
            
            oldf,oldl = self.next_sectioned_train_batch(bsize=50)
            self.livestream.input(newfeats,labels)
            newf,newl = self.livestream.grab_stream_batch(maxkey,batchsize=50)
            print(self.classmap)
            newlbls = np.array(list(map(lambda x:name2lbl_vec(
                self.classmap,x,len(self.classmap.keys())),newl)))
            return np.concatenate((oldf,newf)),np.concatenate((oldl,newlbls)),self.classmap

        
        
        
    
    #This method is for loading data and putting it in non random order
    #sorted/grouped by the class type of the feature
    def load_sorted_features(self):
        if self.features is None or self.labels is None:
            fnames = sorted(self.queryfiles())
            clsmap = json.loads(open('classmap.json').read())
            numkeys = len(list(clsmap.keys()))
            mapper = mapper_lbl_vec(clsmap,numkeys)
        if self.features is None:
            self.features = np.array([np.loadtxt(fn) for fn in fnames])
            self.labels =   np.vstack(np.array(list(map(mapper,fnames))))
            from IPython import embed
            embed()
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
            self.currentclss.append(id)
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

                
    #loads the next batch from the current class unless
    #the nextcls flag is passed, in which case it will
    #load samples from the next class, not loading any data from
    #previous classes
    def next_sectioned_train_batch(self,bsize=100,nextcls=False,nclasses=None):
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
        outlen = len(self.classmap.keys()) if nclasses is None else nclasses
        lbls = np.array(list(map(lambda x:name2lbl_vec(
            self.classmap,x,outlen),np.hstack(self.traininglbls[id][idx]))))

        return self.trainingfeats[id][idx],lbls

    #This method is used for the case of training a classifier
    #with a subset of the classes and adding a new class after setting a flag
    #for the purpose of detecting a new class
    def next_novelty_train_batch(self,bsize=100,newclass=False):
        self.load_sectioned_features()
        if newclass:
            self.updateclass()
        
            
        
    #use this method to simulate new class introduction
    #that is intermingled with previous data instead of
    #looked at in a solitary fashion
    def next_time_mixed_train_batch(self,bsize=100,nextcls=False,nclasses=None):
        self.load_sectioned_features()
        if self.currentcls is None:
            self.updateclass()
        if nextcls:
            self.updateclass()
        clscountsarr = []
        clscounts = {}
        chunks = chunker(bsize,len(self.currentclss))
        for cls in self.currentclss:
            maxi = len(self.trainingfeats[cls])
            clscountsarr.append(maxi)
            clscounts[cls] = maxi
        
        trainingfeats = None
        traininglbls = None
        for chunksize,cls in zip(chunks,self.currentclss):
            
            maxi = clscounts[cls]
            idx = np.mod(np.arange(self.mixedindxs[cls],self.mixedindxs[cls]+chunksize),maxi)
            self.mixedindxs[cls]  = self.mixedindxs[cls] + chunksize % maxi
            trainingfeats = self.trainingfeats[cls][idx] if trainingfeats is None else np.vstack([trainingfeats,self.trainingfeats[cls][idx]])
            traininglbls  = self.traininglbls[cls][idx] if traininglbls is None else  np.vstack( [traininglbls ,self.traininglbls[cls][idx]])

        outlen = len(self.classmap.keys()) if nclasses is None else nclasses
        lbls = np.array(list(map(lambda x:name2lbl_vec(
            self.classmap,x,outlen),np.hstack(traininglbls))))

        return trainingfeats,lbls


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


    def get_sectioned_testing_lbls(self,nclasses=None):
        outlen = len(self.classmap.keys()) if nclasses is None else nclasses
        lbls = np.array(list(map(lambda x:name2lbl_vec(
            self.classmap,x,outlen),np.hstack(self.sectionedtestinglbls))))

        return lbls
#This will be for the handling of a class of streamed data instead of cached data
class datagenstream:
    def __init__(self):
        self.features = {}
        self.labels = set([])
        self.mixedindxs={}

    def input(self,features,labels=None):
        if labels is not None:
            for lbl in set(list(labels)):
                self.features[lbl] = features[labels==np.array([lbl])]
                self.labels = self.labels.union([lbl])
                if lbl not in self.mixedindxs:
                    self.mixedindxs[lbl] = 0
        else:
            print('What Am I Looking at?')    
                
    def grab_stream_batch(self,labelsize,batchsize=100):
        chunks = chunker(batchsize,len(list(self.labels)))
        feats = None
        lbls = None
        for chunksize,label in zip(chunks,list(self.labels)):
            maxi = len(self.features[label])
            idx = np.mod(np.arange(self.mixedindxs[label],self.mixedindxs[label]+chunksize),maxi)
            self.mixedindxs[label] = (self.mixedindxs[label]+chunksize) %maxi
            newfeats = self.features[label][idx]
            print(labelsize)
            newlbls  = np.array(list(map(lambda x: label,newfeats)))
            if feats is not None and lbls is not None:
                feats = np.vstack([feats,newfeats])
                lbls = np.vstack([lbls,newlbls])
            else:
                feats = newfeats
                lbls = newlbls
        return feats,lbls

    

            
# based on http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
def chunker(num,numparts):
    factor = num/numparts
    return [ int(round(factor*(x+1)) - round(factor * x)) for x in range(numparts)]

def mapper_name(fname):
    return fname[fname.rfind('/')+1:fname.find('_')]

def name2lbl_vec(nmmap,name,length):
    return lbl_vec(nmmap[name],length)
            
def mapper_lbl_vec(nmmap,length):

    def fun(name):
        name = name[name.rfind('/')+1:name.find('_')]
        cls = nmmap[name]
        lvec = lbl_vec(cls,length)
        return lvec
    return fun

def lbl_vec(ix,length):
    lvec = np.zeros((length,),dtype=np.uint8)
    lvec[ix] = 1
    return lvec

if __name__ == '__main__':
    datagen = datagenerator(sorteddata=True)
    # for i in range(1,1000):
    #     data = datagen.next_time_mixed_train_batch(100,i % 100 == 0)
    #     print (data[0].shape,data[1].shape)

    # feats = np.random.sample((1000,128))
    # lbls = list(np.repeat(np.array('test'),1000))
    # datagen.livefeed(feats,labels=lbls)
        
    # datagen = datagenstream()
    # feats = np.random.sample((1000,1))
    # lbls = np.random.randint(3,size=(1000,))
    # datagen.input(feats,labels=lbls)
    # for i in range(0,10):
    #     data,lbls = datagen.grab_stream_batch(labelsize=3)
    #     print (len(data),len(lbls))
        
    datagen.load_sorted_features()    
