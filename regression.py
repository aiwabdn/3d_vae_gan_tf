import numpy as np
import tensorflow as tf

# generate data
X_input = np.linspace(-1,1,100)
y_input = X_input * 4 + np.random.randn(X_input.shape[0]) * 0.5

X = tf.placeholder(tf.float32, name='X')
y = tf.placeholder(tf.float32, name='y')

w = tf.Variable(0., name='weight')
b = tf.Variable(0., name='bias')

y_pred = X * w + b
loss = tf.square(y - y_pred, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        _, loss_val = sess.run([optimizer,loss], feed_dict={X:X_input, y:y_input})
        w_val, b_val = sess.run([w,b])
        print(np.mean(loss_val),',',w_val,',',b_val)


############## logistic regression
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('~/learning/tf/data/mnist/', one_hot=True)

learning_rate = 0.1
batch_size = 128
n_epochs = 25
X = tf.placeholder(tf.float32, [batch_size, 784])
y = tf.placeholder(tf.float32, [batch_size, 10])
w = tf.Variable(tf.random_normal(shape=[784,10], stddev=0.01), trainable=True, name='weight')
b = tf.Variable(tf.zeros([1,10]), name='bias')
logits = tf.matmul(X,w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs):
        for _ in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            _,loss_val = sess.run([optimizer,loss], feed_dict={X:X_batch, y:y_batch})
        print(loss_val)
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, y_batch = mnist.test.next_batch(batch_size)
        _,loss_val, logits_val = sess.run([optimizer, loss, logits], feed_dict={X:X_batch, y:y_batch})
        preds = tf.nn.softmax(logits_val)
        correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)
    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
    writer.close()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    writer.close()
