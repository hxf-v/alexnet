import glob
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

"""
预测多张图片，返回图片标签
"""
cornNum = 2  # 待预测的病害标签
y_pre = []  # 定义一个空list用于存放预测结果

model = load_model(r".\AlexNet.h5")  # 加载训练好的模型参数
fileIn = glob.glob(r'E:\alexnet\maize_data\val\2\*.jpg')  # 使用glob匹配文件下的所有图片

try:
    for jpgFile in fileIn:
        img = image.load_img(jpgFile, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        y = model.predict_classes(x)  # 单张图片预测，返回图片标签
        y_pre.append(int(y[0]))  # 将预测结果加入list
except IOError:
    print("Error: 没有找到文件或读取文件失败")

acc = y_pre.count(cornNum) / len(y_pre)

print('预测出的标签结果为：{}'.format(y_pre))
print('{}号病被预测的图片总数为：{}'.format(cornNum, len(y_pre)))
print('正确检测出{}号病的图片总数为：{}'.format(cornNum, y_pre.count(cornNum)))
print('因此属于{}号病的预测准确率是：{}'.format(cornNum, acc))
