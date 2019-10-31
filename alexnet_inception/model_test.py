import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K, Input, Model

img_width, img_height = 224, 224
# train_data_dir = './corn_data/train/'
# validation_data_dir = './corn_data/val/'
# nb_train_samples = 3484
# nb_validation_samples = 870
# epochs = 10
# batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def simpleconv3(input_shape=input_shape, classes=4):
    img_input = Input(shape=input_shape)
    bn_axis = 3
    x = Conv2D(12, (3, 3), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = Conv2D(24, (3, 3), strides=(2, 2), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv2')(x)
    x = Activation('relu')(x)
    x = Conv2D(48, (3, 3), strides=(2, 2), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv3')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(1200, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input, x)
    return model


model = simpleconv3()
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
# )
#
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')
#
# history_ft = model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)
#
# model.save_weights('model1.h5')
