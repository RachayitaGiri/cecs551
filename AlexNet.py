from __future__ import print_function
import sys
sys.path.append('../')
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from statistics import mean
from scripts.coco_dataset import load_data_subset
import time


##call load data

x_train, x_test, y_train, y_test = load_data_subset()

## Params
#classes= 91
#power= random.uniform(-6,-2)
#lr_rate= 10 ** power
#batch_size= 128
#epochs= 5
#input_shape= 224, 224, 3
#print('Lr rate is :', lr_rate)

# CNN using Keras' Sequential capabilities

model_alexnet = Sequential()


##1.  11x11 convolution with 4x4 stride and 96 filters
model_alexnet.add(Conv2D(96, (11, 11), strides = (4,4), activation = 'relu',
                 input_shape=(224,224,3)))

model_alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

##2.  5x5 convolution with 1x1 stride and 256 filters
model_alexnet.add(ZeroPadding2D((2,2)))
model_alexnet.add(Conv2D(256, (5, 5), strides = (1,1), padding='same', activation = 'relu'))

model_alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

##3.  3x3 convolution with 1x1 stride and 384 filters
model_alexnet.add(ZeroPadding2D((1,1)))
model_alexnet.add(Conv2D(384, (3, 3), strides = (1,1), activation = 'relu'))

##4.  3x3 convolution with 1x1 stride and 384 filters
model_alexnet.add(ZeroPadding2D((1,1)))
model_alexnet.add(Conv2D(384, (3, 3), strides = (1,1), activation = 'relu')) 

##5.  3x3 convolution with 1x1 stride and 256 filters
model_alexnet.add(ZeroPadding2D((1,1)))
model_alexnet.add(Conv2D(256, (3, 3), strides = (1,1), activation = 'relu')) 

model_alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))


## 6. Flatten and dense
model_alexnet.add(Flatten())
model_alexnet.add(Dense(4096, activation = 'relu'))
model_alexnet.add(Dropout(0.5))

#7.  dense
model_alexnet.add(Dense(4096, activation = 'relu'))
model_alexnet.add(Dropout(0.5))

#8.  number of classes
model_alexnet.add(Dense(91, activation = 'sigmoid'))

# Open the file to which the output will be written
resfile = open("../outputs/alexnet_output_test.txt","a")
resfile.write("\n- - - - - - - - - - - - \nAlexNet MODEL EXECUTION DETAILS |\n- - - - - - - - - - - -\n")

# Compile the model
model_alexnet.compile(
    loss='categorical_crossentropy',
    optimizer = Adam(lr=1e-4),
    metrics=["accuracy"]
)

model_alexnet.summary()
model_alexnet.summary(print_fn=lambda x: resfile.write(x + '\n'))


# Train the model for the given number of epochs
history = model_alexnet.fit(
    x_train, y_train,
    steps_per_epoch=10,
    epochs=2,
    verbose=1,
    validation_data=(x_test, y_test),
    validation_steps=10
    #callbacks=[tensorboard]
)


## Tensorboard
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#model.fit(x_train, y_train, verbose=1, callbacks=[tensorboard])

# Write the results to a file
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

resfile.write("\nTraining Losses:\n" + str(loss))
resfile.write("\nTraining Accuracies:\n" + str(acc))
resfile.write("\nValidation Losses:\n" + str(val_loss))
resfile.write("\nValidation Accuracies:\n" + str(val_acc))
resfile.write("\nMean Training Loss = "+str(mean(loss)))
resfile.write("\nMean Validation Loss = "+str(mean(val_loss)))
resfile.write("\nMean Training Accuracy = "+str(mean(acc)))
resfile.write("\nMean Validation Accuracy = "+str(mean(val_acc)))
resfile.write("\nNumber of epochs, steps per epoch = "+str(len(loss))+", 10")
resfile.write("\nTime taken = %s seconds" % duration)    
resfile.write("\nLearning Rate = 1e-4")
resfile.write("\nOptimizer = Adam\n")   
