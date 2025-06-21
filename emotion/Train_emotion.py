import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd

from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, Input
from sklearn.model_selection import train_test_split

# Chargement des images et étiquettes
dataset_folder = "inputE/CK+48" 
sub_folders = os.listdir(dataset_folder)

images = []
labels = []
temp = sub_folders

for sub_folder in sub_folders:
    sub_folder_index = temp.index(sub_folder)
    label = sub_folder_index

    if label in [4, 6]:
        new_label = 0
    elif label in [0, 5]:
        new_label = 1
    else:
        new_label = 2

    path = dataset_folder + '/' + sub_folder
    sub_folder_images = os.listdir(path)

    for image in sub_folder_images:
        image_path = path + '/' + image
        print(image_path + "\t" + str(new_label))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48))
        images.append(image)
        labels.append(new_label)

# Normalisation des images
images_x = np.array(images) / 255
labels_y = np.array(labels)

# Codage des étiquettes en one-hot
num_of_classes = 3
labels_y_encoded = tf.keras.utils.to_categorical(labels_y, num_classes=num_of_classes)

# Division des données en train et test
X_train, X_test, Y_train, Y_test = train_test_split(images_x, labels_y_encoded, test_size=0.25, random_state=10)

# Construction du modèle CNN
input_layer = Input(shape=(48, 48, 1))

conv1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(input_layer)
conv1 = Dropout(0.1)(conv1)
conv1 = Activation('relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
conv2 = Dropout(0.1)(conv2)
conv2 = Activation('relu')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool2)
conv3 = Dropout(0.1)(conv3)
conv3 = Activation('relu')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool3)
conv4 = Dropout(0.1)(conv4)
conv4 = Activation('relu')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

flatten = Flatten()(pool4)

dense_1 = Dense(128, activation='relu')(flatten)
drop_1 = Dropout(0.2)(dense_1)

output = Dense(3, activation="sigmoid")(drop_1)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer="adam", loss=["categorical_crossentropy"], metrics=['accuracy'])

# Enregistrement des meilleurs modèles
fle_s = "output/emotion_model.keras"
checkpointer = ModelCheckpoint(fle_s, monitor='loss', verbose=1, save_best_only=True,
                               save_weights_only=False, mode='auto', save_freq='epoch')
callback_list = [checkpointer]

# Entraînement du modèle
save = model.fit(X_train, Y_train, batch_size=32, validation_data=(X_test, Y_test), epochs=50, callbacks=[callback_list])

# Plot des résultats
train_accuracy = save.history['accuracy']
test_accuracy = save.history['val_accuracy']

fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
ax = ax.ravel()

# Courbe de précision (accuracy)
ax[1].plot(train_accuracy, label='Train Accuracy', color='royalblue', marker='o', markersize=5)
ax[1].plot(test_accuracy, label='Test Accuracy', color='orangered', marker='o', markersize=5)

ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Accuracy', fontsize=14)
ax[1].legend(fontsize=14)
ax[1].tick_params(axis='both', labelsize=12)

fig.suptitle(x=0.5, y=0.92, t="Lineplots showing loss and accuracy of CNN model by epochs", fontsize=16)
