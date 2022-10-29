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
inputX = temp_x[:, 0: 10]

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
n_hidden_1 = 2
n_hidden_2 = 2
learning_rate = 0.01
training_epochs = 2000
display_step = 50
n_samples = len(temp)
batch_size = len(temp)
total_batch = int(n_samples/batch_size)

n_input = 10
n_classes = 2

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


def multilayer_perceptron(inp):
    layer1 = tf.add(tf.matmul(inp, weights['h1']), biases['b1'])
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    output = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return output


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

_inputX = tf.cast(inputX, tf.float32)
evidence = multilayer_perceptron(_inputX)
y_ = tf.nn.softmax(evidence)
pred = tf.nn.softmax(evidence)
cost = tf.reduce_sum(tf.pow(y-y_, 2)) / (2*n_samples)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

avg_set = []
epoch_set = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x = inputX
            batch_y = inputY
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            avg_set.append(avg_cost)
            epoch_set.append(epoch+1)
    print("training phase finished")
    last_result = sess.run(y_, feed_dict={x: inputX})
    training_cost = sess.run(cost, feed_dict={x: inputX, y: inputY})
    print("training cost=", training_cost)
    print("last result=", last_result)
    plt.plot(epoch_set, avg_set, 'o', label='MLP Training phase')
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    result = sess.run(y_, feed_dict={x: x_test})
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(inputY, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))

