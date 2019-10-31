import os
import random
import numpy as np
# 单张图片预测
from keras.preprocessing import image
from keras.models import load_model

model = load_model(r".\modelH5\alexnet1.h5")
file_path = r'44.jpg'

img = image.load_img(file_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
y = model.predict_classes(x)
print(y)


