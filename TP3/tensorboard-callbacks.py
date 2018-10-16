#********************************************
#* Name : AIT MANSOUR Mohamed               *
#* Formation : M2-ILSEN                     *
#* Group : Alternance                       *
#* UCE : Outils pour l'apprentissage Auto.  *
#* TP : 3                                   *
#********************************************

# import libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
# import mnist dataset
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from time import gmtime, strftime

# Load mnist dataset (the data, shuffled and split between train and test sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# format data ( Reshape Gives a new shape to an array without changing its data.)
# This two lines mean that we create mini batches from data 
# Dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. 
# This dataset can be used as a drop-in replacement for MNIST.

#Train on 60000 samples, validate on 10000 samples
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Create neural network
# 28*28 pixels = 784 = number of inputs

model = Sequential()

# It means 784 input parameters, with 90 neurons in the FIRST hidden layer.
model.add(Dense(90,input_dim=784, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilation
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())),histogram_freq=0, write_graph=True, write_images=True)

# Using batches to get better results
history = model.fit(x_train, y_train,
					batch_size=490,
                    epochs=25,
                    verbose=1,
                    validation_data=(x_test, y_test), callbacks=[tensorboard])


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Test loss: 0.4211650045061563
# Test accuracy: 0.9644
