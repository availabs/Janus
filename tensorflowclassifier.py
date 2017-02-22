import tensorflow as tf
from datagen import datagenerator
import numpy as np
import os
class model:
    def __init__(self,inputshape=128,outputshape=5):
        
        X = tf.placeholder(tf.float32, shape=[None, inputshape])
        Y_ = tf.placeholder(tf.float32, shape=[None,outputshape])

        W = tf.Variable(tf.zeros([inputshape,outputshape]))
        b = tf.Variable(tf.zeros([outputshape]))

        Y = tf.matmul(X,W)+b
        prediction = tf.nn.softmax(Y)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=Y,labels=Y_))
        train_step =tf.train.GradientDescentOptimizer(0.5).minimize(
            cross_entropy)
        correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        init = tf.initialize_all_variables()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)
        self.session = sess
        self.accuracy=accuracy
        self.prediction = prediction
        self.train = train_step
        self.X = X
        self.Y = Y
        self.Y_ = Y_
        self.currdata=None
        self.currlbls=None

    def save(self,filename='tflowmodel.ckpt'):

        saver = tf.train.Saver()
        save_path = saver.save(self.session,
         os.path.join('/home/avail/code/facerecognition/checkpoints/',filename))
        print("Model saved in file %s" % save_path)
    def load_previous(self,
        filename='tflowmodel.ckpt'):

        loader = tf.train.Saver()
        prefix = '/home/avail/code/facerecognition/checkpoints/'
        path = os.path.join(prefix,filename)
        if os.path.exists(path):
            loader.restore(self.session,path)
            print( 'Model Restored' )
        else:
            print( 'CKPT file does not exist' )
    def input_traindata(self,data,lbls):
        self.curr_trdata = data
        self.curr_trlbls = lbls

    def input_data(self,data,lbls=None):
        self.testdata = data
        self.testlbls = lbls
        
    def backprop(self):
        data = {self.X:self.curr_trdata,self.Y_:self.curr_trlbls}
        train = self.train
        self.session.run(train,feed_dict=data)

    def curraccuracy(self):
        testdata = {self.X:self.testdata,self.Y_:self.testlbls}
        return self.session.run(self.accuracy,feed_dict=testdata)

    def trainpred(self):
        data = {self.X: self.curr_trdata,self.Y_:self.curr_trlbls}
        pred = self.prediction
        return self.session.run(pred,feed_dict=data)
        
    
    def non_class_test(self,prediction):
        top2 = prediction.argsort()[:,-2:]
        a = np.arange(prediction.shape[0])
        score = prediction[a,top2[:,1]]/prediction[a,top2[:,0]]
        #score = prediction[a,top2[:,1]] - prediction[a,top2[:,0]]
        return score

    def non_class_window_score(self,prediction,nframes):
        top2 = prediction.argsort()[:,-2:]
        a = np.arange(prediction.shape[0])
        chunks = np.arange(0,prediction.shape[0],nframes)
        score = prediction[a,top2[:,1]]/prediction[a,top2[:,0]]
        x = 0
        while x < (len(chunks)-1):
            t = score[chunks[x]:chunks[x+1]]
            score[chunks[x]:chunks[x+1]] = np.median(t)
            x += 1
        t = score[chunks[x]:]
        score[chunks[x]:] = np.median(t)
                      
        return score
    
    def pred(self):
        pred = self.prediction
        data = { self.X:self.testdata }
        return self.session.run(pred,feed_dict=data)

    def noveltyTest(self,thresh=10):
        gen = datagenerator(sorteddata=True)
        gen.load_sectioned_features()
        self.load_previous()
        thresh = float(input("Input Threshold: "))
        windowsize = float(input("Input windowsize:  "))
        print(" ")
        while thresh > 0:
            for i in range(0,5):
                self.input_data(np.vstack([gen.trainingfeats[i],
                                           gen.testingfeats[i]]))
                size = gen.trainingfeats[i].shape[0];
                preds = self.pred()
                k = self.non_class_window_score(preds,windowsize)
                old = i < 3
                print("Class {}".format("Known" if old else "Unknown"))
                print("Novelty Score stats: Threshold: {}, Windowsize: {}".format(thresh,windowsize))
                print("Mean : {}".format(k.mean()))
                print("Median : {}".format(np.median(k)))
                print("Min : {}".format(k.min()))
                print("Max : {}".format(k.max()))
                print("10% : {}".format(np.percentile(k,10)))
                print("25% : {}".format(np.percentile(k,25)))
                print("75% : {}".format(np.percentile(k,75)))
                print("90% : {}".format(np.percentile(k,90)))
                print("Num below thresh : {}".format(np.sum(k <= thresh)))
                print("Accuracy : {}".format(1*old + (1-2*old)* np.sum(k<=thresh)/k.shape[0]))
                print(" ")
            thresh = float(input("Input Threshold: "))
            windowsize = float(input("Input windowsize: "))
    def test(self):
        generator = datagenerator(sorteddata=True)
        lasty = -1
        for y in range(10):
            for x in range(100):
                newclass = lasty != y
                if newclass:
                    lasty = y
                bx,by = generator.next_time_mixed_train_batch(100,
                                                              nextcls=newclass,nclasses=5)

                
                self.input_traindata(bx,by)
                self.backprop()
            
            testingfeats = generator.sectionedtestingfeats
            testinglbls = generator.get_sectioned_testing_lbls(5)
            self.input_data(testingfeats,testinglbls)
            acc = self.curraccuracy()
            print(testinglbls.shape)
            print( y, acc )
        self.save()
        print (generator.classmap)
        print (self.pred())

def chunker(num,numparts):
    factor = num/numparts
    return [ int(round(factor*(x+1)) - round(factor * x)) for x in range(numparts)]




if __name__ == '__main__':
    nn = model()
    nn.test()
