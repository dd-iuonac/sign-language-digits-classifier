from os import listdir

import numpy as np
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import models, layers
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import seed

#set the seed
seed(2)

# settings:
img_size = 64
grayscale_images = False
num_class = 10
test_size = 0.2


def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path, flatten=grayscale_images)
    img = imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img


def get_dataset(dataset_path='Dataset'):
    # Getting all data from data path:
    try:
        X = np.load('X.npy')
        Y = np.load('Y.npy')
    except:
        labels = sorted(listdir(dataset_path)) # Geting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(i)
        # Create dataset:
        X = 1-np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        np.save('X.npy', X)
        np.save('Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X, X_test, Y, Y_test

# load data set
X_train, X_test, Y_train, Y_test = get_dataset()
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_train shape: ", Y_train.shape)
print("Y_test shape: ", Y_test.shape)

# define the model
model = models.Sequential()
model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=11, batch_size=32)
test_loss, test_accuracy = model.evaluate(X_test, Y_test)

print("Loss: ", test_loss)
print("Accuracy: ", test_accuracy)

# save the model
model.save('Models/Model_accuracy_' + str(round(test_accuracy *100, 2)))

# fit the model to dataset and print error
x_val = X_train[:1000]
partial_x_train = X_train[1000:]
y_val = Y_train[:1000]
partial_y_train = Y_train[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=11, batch_size=32, validation_data=(x_val, y_val))

acc = history.history["acc"]
loss = history.history["loss"]
val_acc = history.history["val_acc"]
val_loss = history.history["val_loss"]
epochs = (range(1, len(acc) +1 ))

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.title("Trainning and validation loss")
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
plt.show()
