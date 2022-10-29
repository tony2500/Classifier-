import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
np.set_printoptions(threshold=np.inf)
cancer = datasets.load_breast_cancer()
temp_x = cancer.data
temp = cancer.target
inputX = temp_x[:, 0: 30]

inputY = []

for i in range(len(temp)):
    if temp[i] == 0:
        inputY.append([0., 1.])
    else:
        inputY.append([1., 0.])

scaler = StandardScaler()
inputX = scaler.fit_transform(inputX)
inputX = scaler.transform(inputX)

print(inputX)
print(inputY)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(inputX, inputY, test_size=0.4, random_state=0,shuffle=True,stratify=inputY)

yc = cancer.target

learning_rate = 0.01
training_epochs = 2000
display_step = 50
n_samples = len(temp)
batch_size = len(temp)
total_batch = int(n_samples/batch_size)

n_input = 30
n_classes = 2

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

w = tf.Variable(tf.zeros([n_input, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

evidence = tf.add(tf.matmul(x, w), b)
y_ = tf.nn.sigmoid(evidence)

cost = tf.reduce_sum(tf.pow(y-y_, 2)) / (2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


avg_set = []
epoch_set = []
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epochs):
        sess.run(optimizer, feed_dict={x: x_train, y: y_train})
        if i % display_step == 0:
            c = sess.run(cost, feed_dict={x: x_train, y: y_train})
            print("Epoch:", '%04d' % (i), "cost=", "{:.9f}".format(c))
            avg_set.append(c)
            epoch_set.append(i+1)
    print("training phase finished")
    w = w.assign(sess.run(w))
    b = b.assign(sess.run(b))
    training_cost = sess.run(cost, feed_dict={x: x_train, y: y_train})
    print("training cost=", training_cost, "\nw=", sess.run(w), "\nb=", sess.run(b))
    last_result = sess.run(y_, feed_dict={x: inputX})
    print("last result=", last_result)
plt.plot(epoch_set, avg_set, 'o', label='SLP Training phase')
plt.ylabel('cost')
plt.xlabel('epochs')
plt.legend()
plt.show()

yc = last_result[:, 0]


init = tf.global_variables_initializer()
with tf.Session() as sess:
      sess.run(init)
      for i in range(training_epochs):
            sess.run(optimizer, feed_dict = {x: inputX, y: inputY})
      pred = tf.nn.softmax(evidence)
      result = sess.run(pred, feed_dict = {x: x_test})

      correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_test, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
