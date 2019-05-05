# This fine-tuning code was taken from Keras documentation and modified slightly to have only
# 91 dense neurons for the COCO dataset

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
import pprint

# dimensions of our images.
img_width, img_height = 224, 224

epochs = 1
batch_size = 16

# Load our data
x_train, x_test, y_train, y_test = load_data_subset(10)

nb_train_samples = x_train.shape[0]                          
nb_validation_samples = x_test.shape[0]                      

# Build the VGG16 network
input_tensor = Input(shape=(224,224,3))
base_model = VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
# Add an additional MLP model at the "top" (end) of the network
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(91, activation='sigmoid'))
model = Model(input= base_model.input, output= top_model(base_model.output))

#pprint.pprint(model.layers)
NUM_LAYERS = 1

# Freeze all the layers in the original model (fine-tune only the added Dense layers)
for layer in model.layers[:NUM_LAYERS]:       # You need to figure out how many layers were in the base model to freeze
    layer.trainable = False

# Compile the model with a SGD/momentum optimizer and a slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

model.summary()

# Fine-tune the model
model.fit
    x_train, y_train
    samples_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,				            # For Keras 2.0 API change to epochs=epochs,
    validation_data=(x_test, y_test),
    validation_steps=nb_validation_samples//batch_size)       # For Keras 2.0 API change to validation_steps=nb_validation_samples

model.save("vgg_transfer_model.h5")
