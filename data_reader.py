import numpy as np
from matplotlib import image, pyplot as plt
from PIL import Image
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = ( x_train[0:1000].reshape(1000, 28*28) /255, y_train[0:1000]  )

pom = np.zeros((len(labels), 10))
for i, k in enumerate(labels):
    pom[i][k] = 1
labels = pom


test_images=  x_test[0:1000].reshape(1000, 28*28) /255
test_labels = np.zeros((len(y_test), 10))
for i, k in enumerate(y_test):
    test_labels[i][k] = 1

print("Data_loaded")



