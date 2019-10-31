from keras import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model

img_width, img_height = 224, 224
nb_train_samples = 3484
nb_validation_samples = 870
epochs = 10
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model1 = Input(shape=input_shape)
x = Conv2D(96, kernel_size=(7, 7), activation='relu', padding='same', kernel_initializer='uniform')(model1)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_initializer='uniform')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_initializer='uniform')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)

output = concatenate([tower_1, tower_2, tower_3, tower_4], axis=1)


x = GlobalAveragePooling2D()(output)
x = Dense(4, activation='softmax')(x)
model = Model([model1], [x])

model.summary()

plot_model(model, to_file='newModel.png', show_shapes=True, show_layer_names=True)


BatchNormalization