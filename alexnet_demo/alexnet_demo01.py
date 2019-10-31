import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 224, 224
train_data_dir = 'E:/alexnet/maize_data/train/'
validation_data_dir = 'E:/alexnet/maize_data/val/'
nb_train_samples = 4374
nb_validation_samples = 660
epochs = 150
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('AlexNet.h5')

train_loss = history_ft.history['loss']
val_loss = history_ft.history['val_loss']
train_acc = history_ft.history['accuracy']
val_acc = history_ft.history['val_accuracy']
a = str(val_acc) + '\n'
b = str(train_acc) + '\n'
c = str(train_loss) + '\n'
d = str(val_loss) + '\n'
with open('val_acc.txt', 'w') as f:
    f.write(a)
with open('train_acc.txt', 'w') as f:
    f.write(b)
with open('train_loss.txt', 'w') as f:
    f.write(c)
with open('val_loss.txt', 'w') as f:
    f.write(d)
