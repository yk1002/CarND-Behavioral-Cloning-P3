#!/usr/bin/env python

import os
import csv
import cv2
#import PIL import Image
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, AveragePooling2D
from keras import optimizers
import tensorflow as tf
from random import random

# load sample csv
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) # skip the header line
    for line in reader:
        samples.append(line)

# split images samples into trainining and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# remove 90% of samples with 0 steering
fixed_train_samples = []
for train_sample in train_samples:
    if float(train_sample[3]) == 0.0 and random() > 0.1:
        continue
    fixed_train_samples.append(train_sample)
train_samples = fixed_train_samples

# image sample genrator
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)

        for offset in range(0, num_samples, int(batch_size/3)):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            angle_offset = 0.35
            base_dir ='./data/IMG/'
            for batch_sample in batch_samples:
                center_image_file = base_dir + batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(center_image_file), cv2.COLOR_BGR2RGB) 
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                left_image_file = base_dir + batch_sample[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(left_image_file), cv2.COLOR_BGR2RGB) 
                left_angle = center_angle + angle_offset
                images.append(left_image)
                angles.append(left_angle)

                right_image_file = base_dir + batch_sample[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(right_image_file), cv2.COLOR_BGR2RGB) 
                right_angle = center_angle - angle_offset
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_sample_size = len(train_samples) * 3
validation_sample_size = len(validation_samples) * 3
print("train_sample_size = ", train_sample_size)
print("validation_sample_size = ", validation_sample_size)
batch_size = 30
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
input_shape = (160, 320, 3)

#
# Build CNN
#
model = Sequential()
model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=input_shape))
#model.add(AveragePooling2D(pool_size=(2,2), strides=2))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#
# Train model
#
epoch = 5
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_sample_size/batch_size,
                    epochs=epoch,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=validation_sample_size/batch_size)

#
# Save model
#
model.save('model.h5')
