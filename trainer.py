import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_image_data():
    faces_ = []
    labels_ = []
    names_ = {}
    folders = os.listdir("sample/")
    for index, folder in enumerate(folders):
        names_[index] = folder
        for sample in os.listdir("sample/" + folder):
            faces_.append(cv2.imread('sample/' + folder + '/' + sample, cv2.IMREAD_GRAYSCALE))
            labels_.append(index)
    return [np.array(faces_), np.array(labels_), names_]


[samples, labels, names] = get_image_data()
samples = samples / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 50)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(samples, labels, epochs=10)
model.save("models/face_model")
