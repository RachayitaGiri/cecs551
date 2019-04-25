# VGGNet 16 Layer Implementation
vggModel = Sequential()

# Convolution with 64 filters of size 3x3
vggModel.add(ZeroPadding2D((1,1)))
vggModel.add(Conv2D(64, (3,3), strides = (1,1), input_shape=x_train.shape[1:]))
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

vggModel.summary()