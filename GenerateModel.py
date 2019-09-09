# Just disables the warning, doesn't enable AVX/FMA
import os

import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifications = 4
input_shape = (64, 64, 3)

model = Sequential()
model.add(Conv2D(100, kernel_size=(2, 2), strides=(2, 2), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add((Dense(128, kernel_initializer='uniform', activation='relu')))
model.add((Dense(classifications, activation='softmax')))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip =True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
r'C:\Users\Aneta\PycharmProjects\biai\training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

test_set = train_datagen.flow_from_directory(
r'C:\Users\Aneta\PycharmProjects\biai\test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

STEP_SIZE_TRAIN = training_set.n//training_set.batch_size
STEP_SIZE_VALID = test_set.n//test_set.batch_size

model.fit_generator(
    training_set,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=80,
    validation_data=test_set,
    validation_steps=STEP_SIZE_VALID,
    workers=6,
    callbacks=[tbCallBack]
)
model.save('seasonsAIPostTrain.h5')





