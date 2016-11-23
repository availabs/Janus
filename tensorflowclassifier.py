import tensorflow as tf
from datagen import datagenerator
X = tf.placeholder(tf.float32, shape=[None, 128])
Y_ = tf.placeholder(tf.float32, shape=[None,5])

W = tf.Variable(tf.zeros([128,5]))
b = tf.Variable(tf.zeros([5]))

Y = tf.matmul(X,W)+b
prediction = tf.argmax(tf.nn.softmax(Y),1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y,Y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
generator = datagenerator(sorteddata=True)
lasty = -1
for y in range(10):
    for x in range(1000):
        newclass = lasty != y
        if newclass:
            lasty = y
        bx,by = generator.next_sectioned_train_batch(10,nextcls=newclass)

        sess.run(train_step,feed_dict={X:bx,Y_:by})
    testingfeats = generator.sectionedtestingfeats
    testinglbls = generator.sectionedtestinglbls
    
    correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    acc = sess.run(accuracy,feed_dict=
                   {X:testingfeats,
                    Y_:testinglbls})
    print(testinglbls.shape)
    print(y,acc)
print (sess.run(prediction,feed_dict={X:testingfeats,Y_:testinglbls}))
