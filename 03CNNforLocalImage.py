"""
# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
#
#  Downloaded Cell Images, Malaria Dataset
"""

import os

# Check if filepath is okay
data_dir = "/Users/atanushome/Documents/Tutorials/cell_images"
print(os.listdir(data_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from matplotlib.image import imread

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import load_model

train_path = data_dir + "/train/"
test_path = data_dir + "/test/"
print(len(os.listdir(train_path + "parasitized")))
print(len(os.listdir(train_path + "uninfected")))
print(len(os.listdir(test_path + "parasitized")))
print(len(os.listdir(test_path + "uninfected")))

# Lets observe the shape
dim1 = []  # dimension
dim2 = []

for image_file in os.listdir(test_path + "uninfected/"):

    img = imread(test_path + 'uninfected/' + image_file)
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

# Observe variation in dimension
sns.jointplot(dim1, dim2)
plt.show()  # Different image shapes

# We need to convert all images into one single shape
x_shape = int(np.mean(dim1))
y_shape = int(np.mean(dim2))

image_shape = (x_shape, y_shape, 3)

# We can randomly expand the no_of_images by several transformation
image_gen = ImageDataGenerator(rotation_range=20,  # rotate the image 20 degrees
                               width_shift_range=0.10,  # Shift the pic width by a max of 5%
                               height_shift_range=0.10,  # Shift the pic height by a max of 5%
                               rescale=1/255,  # Rescale the image by normalzing it.
                               shear_range=0.1,  # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1,  # Zoom in by 10% max
                               horizontal_flip=True,  # Allo horizontal flipping
                               fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                               )

batch_size = 20
train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary'
                                                )
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',
                                               shuffle=False
                                               )

# Save multiple Training Time
if os.path.isfile('custom_image_Model.h5'):
    model = load_model('custom_image_Model.h5')
else:

    # Create Model
    model = Sequential()

    # Convolution Layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                     input_shape=image_shape, activation='relu'))
    # padding could be valid / same

    # Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Good idea to choose pool_size to be half of kernel_size

    # Another Convolution Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                     input_shape=image_shape, activation='relu'))
    # Another Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Another Convolution Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                     input_shape=image_shape, activation='relu'))
    # Another Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Flatten the image
    model.add(Flatten())
    # Dense layer
    model.add(Dense(256, activation='relu'))  # 256  neurons
    model.add(Dropout(0.5))

    # OUTPUT Layer
    model.add(Dense(1, activation='sigmoid'))
    # softmax for multiclass classification
    # no_of_classes = no_of_neuron in last layer, here it's 10
    # for multi-class classification, model.add(Dense(10, activation='softmax'))


    # Compile the Model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    # keras.io/metrices

    print(model.summary())

    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    # Since optional metrics['accuracy'] is added, thus we can use val_accuracy


    # Train the Model
    h = model.fit_generator(train_image_gen, epochs=20, validation_data=test_image_gen,
                            callbacks=[early_stop])

    # Model Evaluation
    # Check losses
    # loss vs validation_loss
    loss_df = pd.DataFrame(h.history['loss'])
    ax = loss_df.plot()
    loss_df2 = pd.DataFrame(h.history['val_loss'])
    loss_df2.plot(ax=ax)
    plt.show()

    # Model Evaluation
    # Check accuracy
    # accuracy vs validation_accuracy
    loss_df = pd.DataFrame(h.history['acc'])
    ax = loss_df.plot()
    loss_df2 = pd.DataFrame(h.history['val_acc'])
    loss_df2.plot(ax=ax)
    plt.show()

    model.save('custom_image_Model.h5')


# Prediction
pred = model.predict_generator(test_image_gen)  # It will provide probabilities
predictions = pred > 0.5

# Classification Report
print(classification_report(test_image_gen.classes, predictions))

# Confusion Matrix
print(confusion_matrix(test_image_gen.classes, predictions))

# Visualize Confusion Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(test_image_gen.classes, predictions), annot=True)