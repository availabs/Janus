import tensorflow as tf
from datagen import datagenerator

class model:
    def __init__(self,inputshape=128,outputshape=5):
        
        X = tf.placeholder(tf.float32, shape=[None, inputshape])
        Y_ = tf.placeholder(tf.float32, shape=[None,outputshape])

        W = tf.Variable(tf.zeros([inputshape,outputshape]))
        b = tf.Variable(tf.zeros([outputshape]))

        Y = tf.matmul(X,W)+b
        prediction = tf.argmax(tf.nn.softmax(Y),1)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(Y,Y_))
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

    
        
    def pred(self):
        pred = self.prediction
        data = {self.X:self.testdata }
        return self.session.run(pred,feed_dict=data)
        
    def test(self):
        generator = datagenerator(sorteddata=True)
        lasty = -1
        for y in range(20):
            for x in range(100):
                newclass = lasty != y
                if newclass:
                    lasty = y
                bx,by = generator.next_time_mixed_train_batch(100,
                                                              nextcls=newclass)

                self.input_traindata(bx,by)
                self.backprop()
            testingfeats = generator.sectionedtestingfeats
            testinglbls = generator.get_sectioned_testing_lbls()
            self.input_data(testingfeats,testinglbls)
            acc = self.curraccuracy()
    
            print(testinglbls.shape)
            print( y, acc )
        print (self.pred())


        
if __name__ == '__main__':
    nn = model()
    nn.test()
