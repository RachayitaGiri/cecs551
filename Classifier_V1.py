from __future__ import print_function
import sys
sys.path.append('../')
from scripts.coco_dataset import *
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from statistics import mean
from scripts.coco_dataset import load_data
from scripts.coco_dataset import load_data_subset
import time

start = time.time() 

#X_train, X_test, y_train, y_test = load_data()
X_train, X_test, y_train, y_test = load_data_subset()

    

## Model params
input_shape= 224, 224, 3
classes= 91
batch_size= 200
epochs=1

## model


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides= 1 ,padding= 'same', activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(batch_size, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(classes, activation='sigmoid'))
model.summary()

# Open the file to which the output will be written
resfile = open("../outputs/output_test.txt","a")
resfile.write("\n- - - - - - - - - - - - \nMODEL EXECUTION DETAILS |\n- - - - - - - - - - - -\n")


## Params

for i in range(2):
    power= random.uniform(-6,-2)
    lr_rate= 10 ** power
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=Adam(lr=lr_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False),
              metrics=['accuracy'])
    ## Training & fit
    history= model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    pred = model.predict(X_test)
    pred[pred >= 0.5]=1
    pred[pred<0.5]=0
    
    score= f1_score(y_test, pred, average= 'samples')
    print('Test accuracy:', score)
    pred = tf.convert_to_tensor(pred, np.float64)
    loss1 =keras.losses.binary_crossentropy(y_test, pred)
    print('Test loss:', loss1)

    
        




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
        





