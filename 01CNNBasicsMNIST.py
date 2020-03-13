import os.path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix

# Load MNIST data
# Gets downloaded under ~users/tensorflow_datasets
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# Check an image
# single_image = x_train[0]
# plt.imshow(single_image)
# plt.show()
# It's grey image. Why color? Coz matplot is assigning color instead of black and white
# But array shows it's in gray scale
# 0-255 plt.imshow(single_image, cmap='gist_gray')

# MNIST Dataset has number as target y
# It's not efficient to use numbers as target
# So, need to change it in labels
y_cat_train = to_categorical(y_train, num_classes=10)
y_cat_test = to_categorical(y_test, num_classes=10)

# Normalization or scaling is a good idea for gray colored images
x_train = x_train/255
x_test = x_test/255

print(x_train.shape)  # currently (60000,28,28)
# Declare color channel
# 1 for gray scale
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Save multiple Training Time
if os.path.isfile('MNIST_Model.h5'):
    model = load_model('MNIST_Model.h5')
else:

    # Create Model
    model = Sequential()

    # Convolution Layer
    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='valid',
                     input_shape=(28, 28, 1), activation='relu'))
    # padding could be valid / same

    # Pooling Layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Good idea to choose pool_size to be half of kernel_size

    # Flatten the image
    model.add(Flatten())
    # Dense layer
    model.add(Dense(128, activation='relu'))  # 128 neurons

    # OUTPUT Layer
    model.add(Dense(10, activation='softmax'))
    # softmax for multiclass classification
    # no_of_classes = no_of_neuron in last layer, here it's 10
    # for binary classification, model.add(Dense(1, activation='sigmoid'))


    #Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    # keras.io/metrices

    early_stop = EarlyStopping(monitor='val_loss', patience=1)
    # Since optional metrics['accuracy'] is added, thus we can use val_accuracy

    # Train the Model
    h = model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test),
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

    model.save('MNIST_Model.h5')

# Classification Report
predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))

# Confusion Matrix
print(confusion_matrix(y_test, predictions))

# Visualize Confusion Matrix
# plt.figure(figsize=(12, 8))
# sns.heatmap(confusion_matrix(y_test, predictions), annot=True)


# To perform a single prediction
target_image = x_test[0]
plt.imshow(target_image.reshape(28, 28))
plt.show()
result = model.predict_classes(target_image.reshape(1, 28, 28, 1))
print('Target Result ' + str(result))
