







from tensorflow.keras.datasets import fashion_mnist
from matplotlib import pyplot

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# printing train and test data dimensions
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(f'Train: train_images.shape={train_images.shape}, train_labels.shape={test_images.shape}')
print(f'Test: test_images.shape={test_images.shape}, test_labels.shape={test_labels.shape}')

# # print out the imaages to view them
# for i in range(9):
# 	pyplot.subplot(330 + 1 + i)
# 	pyplot.imshow(train_images[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()


# normalize pixel values so they are between -0.5 and +0.5 instead of between 0 and 255
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5


# add third dimension to the images so their "depth" is 1 (because they are greyscale)
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

dict_of_images={}

num=0
for filter1 in range(1,11):
  for filterSize in range(1,6):
    for poolSize in range(2,6):
      for strides in range(1,4):
        for epochs in range(1,8):

          num_filters = filter1
          filter_size = filterSize
          pool_size = poolSize

          # constructing the model- 1 convolutional layer, 1 max pooling layer, 
          # and output layer
          model = Sequential([
            Conv2D(num_filters, filter_size, input_shape=(28, 28, 1), activation="relu", \
            padding="same", strides=strides), 
            MaxPooling2D(pool_size=pool_size),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation='softmax'),
          ])


          # compiling the model
          model.compile(
            'adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
          )

          # training the model
          model.fit(
            train_images,
            to_categorical(train_labels),
            epochs=epochs,

          )
          # testing the model
          history=model.evaluate(
            test_images,
            to_categorical(test_labels)
          )
          dict_of_images[filter1, filterSize,poolSize,strides,epochs]=history
          num+=1
          print("num"+str(num))



print(dict_of_images)