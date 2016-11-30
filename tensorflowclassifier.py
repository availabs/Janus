import tensorflow as tf
from datagen import datagenerator

class model:
    def __init__(self):
        
        X = tf.placeholder(tf.float32, shape=[None, 128])
        Y_ = tf.placeholder(tf.float32, shape=[None,5])

        W = tf.Variable(tf.zeros([128,5]))
        b = tf.Variable(tf.zeros([5]))

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
    def test(self):
        generator = datagenerator(sorteddata=True)
        lasty = -1
        testingfeats = generator.sectionedtestingfeats
        testinglbls = generator.sectionedtestinglbls
        for y in range(20):
            for x in range(1000):
                newclass = lasty != y
                if newclass:
                    lasty = y
                bx,by = generator.next_time_mixed_train_batch(10,
                                                              nextcls=newclass)
        
            self.session.run(self.train,feed_dict={self.X:bx,self.Y_:by})
            testingfeats = generator.sectionedtestingfeats
            testinglbls = generator.sectionedtestinglbls
            acc = self.session.run(self.accuracy,feed_dict=
                   {self.X:testingfeats,
                    self.Y_:testinglbls})

    
            print(testinglbls.shape)
            print( y, acc )
        print (self.session.run(self.prediction,
                                feed_dict={self.X:testingfeats,
                                           self.Y_:testinglbls}))
        
if __name__ == '__main__':
    nn = model()
    nn.test()
