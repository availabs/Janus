# coding: utf-8

import numpy as np
from tensorflowclassifier import model
mymodel = model()
from datagen import datagenerator
dg = datagenerator()
mymodel.load_previous
mymodel.load_previous()
dg.load_sectioned_features()
mymodel.input_data(dg.testingfeats[0],dg.testinglbls[0])
dg.testingfeats
mymodel.input_data(dg.testingfeats[0],dg.testinglbls[0])
mymodel.input_data(dg.trainingfeats[0],dg.traininglbls[0])
dg.traininglbls[0]
mymodel.input_data(dg.trainingfeats[2],dg.traininglbls[2])
dg.traininglbls[2]
p =mymodel.pred()
p
np.sum(p.argmax() == 2)/p.shape[0]
p
p.argmax()
p.argmax(axis=1)
mymodel.input_data(dg.trainingfeats[3],dg.traininglbls[3])
p = mymodel.pred()
p
p.argmax(axis=1)
p.argmax(axis=1) == 3
np.sum(p.argmax(axis=1) == 3)/p.shape[0]
p.argmax(axis=1)
p.argsort()[-2:]
p.argsort(axis=1)[-2:]
p.argsort(axis=0)[-2:]
p.argsort(axis=0)
p.argsort(axis=1)
p.argsort(axis=1)[:,-2:]
ix = p.argsort(axis=1)[:,-2:]
p[ix]
p[ix]
p
ix
p.shape
a = np.arange(p.shape(0))
a = np.arange(p.shape[0])
a
p[a,ix]
ix[:,1]
p[a,ix[:,1]]
p[a,ix[:,0]]/p[a,ix[:,1]]
p[a,ix[:,1]]
p[a,ix[:,1]]/p[a,ix[:,0]]
r = p[a,ix[:,1]]/p[a,ix[:,0]]
np.min(r)
np.max(r)
mymodel.input_data(dg.trainingfeats[1],dg.traininglbls[1])
p = mymodel.pred()
ix = p.argsort(axis=1)[:,-2:]
ix
p.shape
a = np.arange(p.shape[0])
a
p[a,ix[:,0]]
p[a,ix[:,1]]/p[a,ix[:,0]]
t = p[a,ix[:,1]]/p[a,ix[:,0]]
np.min(t)
np.max(t)
np.median(t)
np.mean(t)
