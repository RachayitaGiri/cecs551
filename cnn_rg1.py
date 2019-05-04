from __future__ import print_function
import sys
sys.path.append('../')
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from statistics import mean
from scripts.coco_dataset import load_data
import time

start = time.time() 

x_train, x_test, y_train, y_test = load_data()

#num_pixels = x_train.shape[0] * x_train.shape[1]               # guess don't need it for CNNs

# Train the model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

""" model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25)) """

model.add(Flatten())
model.add(Dense(1024, activation='relu', ))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(91, activation='sigmoid'))

# Open the file to which the output will be written
resfile = open("../outputs/output_test.txt","a")
resfile.write("\n- - - - - - - - - - - - \nMODEL EXECUTION DETAILS |\n- - - - - - - - - - - -\n")

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer = Adam(lr=1e-4),
    metrics=["accuracy"]
)

mystr = model.summary()
model.summary(print_fn=lambda x: resfile.write(x + '\n'))

# Train the model for the given number of epochs
history = model.fit(
    x_train, y_train,
    steps_per_epoch=10,
    epochs=2,
    verbose=1,
    validation_data=(x_test, y_test),
    validation_steps=10
)

duration = time.time() - start

# Write the results to a file
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

resfile.write("\nTraining Losses:\n" + loss)
resfile.write("\nTraining Accuracies:\n" + acc)
resfile.write("\nValidation Losses:\n" + val_loss)
resfile.write("\nValidation Accuracies:\n" + val_acc)
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
