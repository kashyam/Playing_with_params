# Simple CNN model for the CIFAR-10 Dataset
import numpy
from keras import callbacks
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), input_shape=(32, 32,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), input_shape=(32, 32,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(1024, (2, 2), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
seqModel = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=256, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Train Accuracy: %f" % (scores[1]*100))
print("Train loss: %f" % (scores[0]))

# visualizing losses and accuracy
train_loss = seqModel.history['loss']
val_loss = seqModel.history['val_loss']
train_acc = seqModel.history['acc']
val_acc = seqModel.history['val_acc']
xc = range(epochs)

# Plotting loss
plt.figure()
plt.plot(xc, train_loss, label='Train loss')
plt.plot(xc, val_loss, label='Val loss')
plt.legend()

# Plotting Accuracy
plt.figure()
plt.plot(xc, train_acc, label='Train Acc')
plt.plot(xc, val_acc, label='Val Acc')
plt.legend()