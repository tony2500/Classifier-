import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from inspect import stack
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras as ks
np.set_printoptions(threshold=np.inf)
cancer = datasets.load_breast_cancer()
temp_x = cancer.data
temp = cancer.target
inputX = temp_x[:, 0: 30]

inputY = []

for i in range(len(temp)):
    if temp[i] == 0:
        inputY.append([0, 1])
    else:
        inputY.append([1, 0])

scaler = StandardScaler()
inputX = scaler.fit_transform(inputX)
inputX = scaler.transform(inputX)

print(inputX)
print(inputY)
inputY = np.array(inputY).astype('int32')
learning_rate = 0.01
training_epochs = 2000
display_step = 50
n_samples = len(temp)
batch_size = len(temp)
total_batch = int(n_samples/batch_size)

n_input = 30
n_classes = 2

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(inputX, inputY, test_size=0.4, random_state=0, shuffle=True, stratify=inputY)
nn_model=ks.models.Sequential()
nn_model.add(tf.keras.Input(shape=(n_input,)))
nn_model.add(ks.layers.Dense(n_classes, activation='softmax'))
optimizer = 'adam'
loss_function = 'mean_squared_error'
metric = ['accuracy']
nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
nn_model.fit(x_train, y_train, epochs=600)
test_loss,test_accuaracy = nn_model.evaluate(x_test, y_test)
print('Test Data Accuracy {}'.format(round(float(test_accuaracy), 2)))
print('Test loss {}'.format(round(float(test_loss), 4)))


