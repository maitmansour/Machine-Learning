#********************************************
#* Name : AIT MANSOUR Mohamed               *
#* Formation : M2-ILSEN                     *
#* Group : Alternance                       *
#* UCE : Outils pour l'apprentissage Auto.  *
#* TP : 2                                   *
#********************************************

# import libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Create neural network
model = Sequential()
model.add(Dense(200,input_dim=400, activation='relu'))
model.add(Dense(200,input_dim=400, activation='relu'))
model.add(Dense(10,input_dim=400, activation='softmax'))


# Compilation
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])