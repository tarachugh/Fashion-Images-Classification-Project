from tensorflow.keras.datasets import fashion_mnist
from matplotlib import pyplot

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

class Fashion():

  def __init__(self, num_filters, filter_size, pool_size, num_strides, num_epochs):
    self.num_filters=num_filters
    self.filter_size=filter_size
    self.pool_size=pool_size
    self.num_strides=num_strides
    self.num_epochs= num_epochs

  def prepare_data(self):
    # printing train and test data dimensions
    # (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
    # print(f'Train: train_images.shape={self.train_images.shape}, train_labels.shape={self.test_images.shape}')
    # print(f'Test: test_images.shape={self.test_images.shape}, test_labels.shape={self.test_labels.shape}')

    # # print out the imaages to view them
    # for i in range(9):
    # 	pyplot.subplot(330 + 1 + i)
    # 	pyplot.imshow(train_images[i], cmap=pyplot.get_cmap('gray'))
    # pyplot.show()


    # normalize pixel values so they are between -0.5 and +0.5 instead of between 0 and 255
    self.train_images = (self.train_images / 255) - 0.5
    self.test_images = (self.test_images / 255) - 0.5


    # add third dimension to the images so their "depth" is 1 (because they are greyscale)
    self.train_images = np.expand_dims(self.train_images, axis=3)
    self.test_images = np.expand_dims(self.test_images, axis=3)
    
  def create_model(self):
    # constructing the model
    self.model = Sequential([
      Conv2D(self.num_filters, self.filter_size, input_shape=(28, 28, 1), activation="relu", \
      padding="same", strides=self.num_strides), 
      MaxPooling2D(pool_size=self.pool_size),
      Flatten(),
      Dense(64, activation="relu"),
      Dense(10, activation='softmax'),
    ])

  def compile_model(self):
    # compiling the model
    self.model.compile(
      'adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )

  def train_model(self):
    # training the model
    self.model.fit(
      self.train_images,
      to_categorical(self.train_labels),
      epochs=self.num_epochs,
    )

  def evaluate_model(self):
    # testing the model
    self.model.evaluate(
    self.test_images,
    to_categorical(self.test_labels)
    )


# putting everything together:
Model= Fashion(10,5,2,1,7)
Model.prepare_data()
Model.create_model()
Model.compile_model()
Model.train_model()
Model.evaluate_model()




