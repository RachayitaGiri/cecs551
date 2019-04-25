# Let's build a CNN using Keras' Sequential capabilities

model_alexnet = Sequential()


##1.  11x11 convolution with 4x4 stride and 96 filters
model_alexnet.add(Conv2D(96, (11, 11), strides = (4,4),
                 input_shape=x_train.shape[1:]))
model_alexnet.add(Activation('relu'))

model_alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

##2.  5x5 convolution with 1x1 stride and 256 filters
model_alexnet.add(ZeroPadding2D((2,2)))
model_alexnet.add(Conv2D(256, (5, 5), strides = (1,1))), padding=2
model_alexnet.add(Activation('relu'))

model_alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

##3.  3x3 convolution with 1x1 stride and 384 filters
model_alexnet.add(ZeroPadding2D((1,1)))
model_alexnet.add(Conv2D(384, (3, 3), strides = (1,1)))
model_alexnet.add(Activation('relu'))

##4.  3x3 convolution with 1x1 stride and 384 filters
model_alexnet.add(ZeroPadding2D((1,1)))
model_alexnet.add(Conv2D(384, (3, 3), strides = (1,1))) 
model_alexnet.add(Activation('relu'))

##5.  3x3 convolution with 1x1 stride and 256 filters
model_alexnet.add(ZeroPadding2D((1,1)))
model_alexnet.add(Conv2D(256, (3, 3), strides = (1,1))) 
model_alexnet.add(Activation('relu'))

model_alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))


## 6. Flatten and dense
model_alexnet.add(Flatten())
model_alexnet.add(Dense(4096))
model_alexnet.add(Activation('relu'))
model_alexnet.add(Dropout(0.5))

#7.  dense
model_alexnet.add(Dense(4096))
model_alexnet.add(Activation('relu'))
model_alexnet.add(Dropout(0.5))

#8.  number of classes
model_alexnet.add(Dense(91))
model_alexnet.add(Activation('sigmoid'))

model_alexnet.summary()