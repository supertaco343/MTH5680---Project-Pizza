import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

DATA_DIR = 'pizza types'
data = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    batch_size=8
)


BATCH_SIZE = 8
IMG_HEIGHT = 28
IMG_WIDTH = 28

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols=5, figsize=(28,28))
for i, img in enumerate(batch[0][:5]):
    ax[i].imshow(img.astype(int))
    ax[i].title.set_text(batch[1][i])
    
    
data = data.map(lambda x,y: (x/255, y))

train_size = int(len(data)*0.8)
test_size = int(len(data)*0.2)

train_ds = data.take(train_size)
test_ds = data.skip(train_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

hist = model.fit(train_ds, epochs=20, verbose=0)

hist.history['accuracy']

model.predict(test_ds)

