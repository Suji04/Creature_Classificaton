from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Convolution2D(64,2,2, input_shape = (128,128,3), activation="relu"))
model.add(Convolution2D(64,2,2,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,2,2,activation="relu"))
model.add(Convolution2D(64,2,2,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,2,2,activation="relu"))
model.add(Convolution2D(64,2,2,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(output_dim=128, activation="relu"))
model.add(Dense(output_dim=5, activation="softmax"))

model.compile(optimizer = "adam", loss="categorical_crossentropy", metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(128, 128),
        batch_size=5,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(128, 128),
        batch_size=5,
        class_mode='categorical') 

model.fit_generator(
        training_set,
        steps_per_epoch=1670,
        epochs=25,
        validation_data=test_set,
        validation_steps=148)

model.save("model2.h5")



