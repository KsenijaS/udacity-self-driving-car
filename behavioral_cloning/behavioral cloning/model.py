import csv
import cv2

IMAGE_SHAPE = (66, 200, 3)
def read_file(file_name):
    lines = []
    with open('./' + file_name + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def preprocess_image(image):
   # cropped_image = image[70, -25, :]
   # resized_image = cv2.resize(cropped_image, (66, 200), cv2.INTER_AREA)
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    return yuv_image
                
def load_data(data, file_name):
    images = []
    measurements = []
    for line in data:
        for i in range(3):
            source_path = line[i]
            tokens = source_path.split('/')
            imgname = tokens[-1]
            local_path = './' + file_name + '/IMG/' + imgname
            image = cv2.imread(local_path)
            image = preprocess_image(image)
            images.append(image)
        correction = 0.1
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(measurement+correction)
        measurements.append(measurement-correction)

    return (images, measurements)

def augment_data(images, measurements):
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = (measurement) * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)

    return (augmented_images, augmented_measurements)


import numpy as np
from sklearn.model_selection import train_test_split
'''
def load_data(images, measurements):
    X = np.array(images)
    y = np.array(measurements)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
'''
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) # (160,320,3)
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
 #   model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

from numpy import random

def data_generator(X, Y):
    batch_size = 50
    images = np.empty([batch_size, 66, 200, 3])
    measur = np.empty(batch_size)
    while True:
        for i in range(batch_size):
            index= random.choice(len(X),1)
            images[i] = X[index]
            measur[i] = Y[index]

        yield (images, measur)

file_name = "data"
data = read_file(file_name)
images, measurements = load_data(data, file_name)
images, measurements = augment_data(images, measurements)
X = np.array(images)
y = np.array(measurements)

model = build_model()

model.compile(optimizer='adam', loss='mse')
#model.fit_generator(data_generator(X_train, y_train), samples_per_epoch=2000, nb_epoch=5, max_q_size=1, validation_data=data_generator(X_valid, y_valid), nb_val_samples=len(X_valid), callbacks=[],verbose=1, use_multiprocessing=True,workers=6)
model.fit(X, y, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')
