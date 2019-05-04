from __future__ import print_function
import sys
sys.path.append('../')
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, ZeroPadding2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from statistics import mean
from scripts.coco_dataset import load_data_subset
import time

from keras.callbacks import TensorBoard

start = time.time() 

x_train, x_test, y_train, y_test = load_data_subset()

# VGGNet 16 Layer Implementation
vggModel = Sequential()

# Convolution with 64 filters of size 3x3
vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(64, (3,3), strides = (1,1), input_shape=(224,224,3)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(64, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

# 2x2 filter with stride 2
vggModel.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution with 128 filters of size 3x3
vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(128, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(128, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

# 2x2 filter with stride 2
vggModel.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution with 256 filters of size 3x3
vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(256, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(256, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(256, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

# 2x2 filter with stride 2
vggModel.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution with 512 filters of size 3x3
vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(512, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(512, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(512, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

# 2x2 filter with stride 2
vggModel.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution with 512 filters of size 3x3
vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(512, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(512, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(512, (3,3), strides = (1,1)))
vggModel.add(Activation('relu'))

# 2x2 filter with stride 2
vggModel.add(MaxPooling2D(pool_size=(2, 2)))

# Need FC layers (2)
vggModel.add(Flatten())

vggModel.add(Dense(4096))
vggModel.add(Activation('relu'))
vggModel.add(Dropout(0.5))

vggModel.add(Dense(4096))
vggModel.add(Activation('relu'))
vggModel.add(Dropout(0.5))

# Need FC Layer (1 for each class)
vggModel.add(Dense(91))

# change loss function?
vggModel.add(Dense(1000))
vggModel.add(Activation('sigmoid'))

# Open the file to which the output will be written
resfile = open("../outputs/VGG_output_test.txt","a")
resfile.write("\n- - - - - - - - - - - - \nMODEL EXECUTION DETAILS |\n- - - - - - - - - - - -\n")

# Compile the model
vggModel.compile(
    loss='categorical_crossentropy',
    optimizer = Adam(lr=1e-4),
    metrics=["accuracy"]
)

# use tensorboard to visualize our models
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

mystr = vggModel.summary()

vggModel.summary(print_fn=lambda x: resfile.write(x + '\n'))

# Train the model for the given number of epochs
history = vggModel.fit(
    x_train, y_train,
    steps_per_epoch=10,
    epochs=2,
    verbose=1,
    validation_data=(x_test, y_test),
    validation_steps=10,
    #callbacks=[tensorboard]
)

duration = time.time() - start

# Write the results to a file
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

resfile.write("\nMean Training Loss = "+str(mean(loss)))
resfile.write("\nMean Validation Loss = "+str(mean(val_loss)))
resfile.write("\nMean Training Accuracy = "+str(mean(acc)))
resfile.write("\nMean Validation Accuracy = "+str(mean(val_acc)))
resfile.write("\nNumber of epochs, steps per epoch = "+str(len(loss))+", 10")
resfile.write("\nTime taken = %s seconds" % duration)    
resfile.write("\nLearning Rate = 1e-4")
resfile.write("\nOptimizer = Adam\n")   

""" # Evaluate the losses of the model 
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, color='red', label='Training Loss')
plt.plot(epochs, val_loss, color='blue', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Evaluate the accuracy of the model
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
 """