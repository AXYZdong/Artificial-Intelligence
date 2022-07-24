## 环境说明
>**操作系统：Windows 10** 
> \
> **CUDA 版本为: 10.0**
> \
> **cudnn 版本为: 7.6.5**
> \
> **Python 版本为：Python 3.6.13**	
> \
> **tensorflow-gpu 版本为：1.13.1**
> \
> **keras	版本为：2.2.4**
> \
> **注意CUDA、cudnn、Python、tensorflow版本之间的匹配**


## 一、模型训练
### 1.1 导入相关依赖

```python
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from tensorflow.examples.tutorials.mnist import input_data
```
### 1.2 数据集导入并编码

```python
# 独热编码
data_folder = "./MNIST_data"
mnist = input_data.read_data_sets(data_folder, one_hot=True)
```
### 1.3 CNN网络搭建
```python
# 搭建CNN卷积神经网络
model = Sequential()
inputShape = (28, 28, 1)  # 输入数据的维度
# 第一层：卷积层+池化层
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=inputShape))  # 卷积层
model.add(MaxPooling2D(pool_size=(2, 2)))  # 最大池化
# 第二层：卷积层+池化层
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 第三层：卷积层+池化层
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 第四层：卷积层+池化层
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 将数据展开成一维
model.add(Dense(1024, activation='relu'))  # 全连接网络
model.add(Dropout(0.5))  # Dropout处理，防止过拟合
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout处理，防止过拟合
model.add(Dense(10, activation='softmax'))  # 分类
```
### 1.4 进行训练

```python
# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(mnist.train.images.reshape((-1, 28, 28, 1)), mnist.train.labels, batch_size=64, epochs=5, verbose=1)
```
### 1.5 模型保存

```python
# 保存模型
model.save('model.h5')
```


### 1.6 完整代码
```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/3/13 20:12
# @Author: AXYZdong
# @File  : Keras.py
# @IDE   : Pycharm
# ===============================
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from tensorflow.examples.tutorials.mnist import input_data
# 独热编码
data_folder = "./MNIST_data"
mnist = input_data.read_data_sets(data_folder, one_hot=True)
# 搭建CNN卷积神经网络
model = Sequential()
inputShape = (28, 28, 1)  # 输入数据的维度
# 第一层：卷积层+池化层
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=inputShape))  # 卷积层
model.add(MaxPooling2D(pool_size=(2, 2)))  # 最大池化
# 第二层：卷积层+池化层
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 第三层：卷积层+池化层
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 第四层：卷积层+池化层
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 将数据展开成一维
model.add(Dense(1024, activation='relu'))  # 全连接网络
model.add(Dropout(0.5))  # Dropout处理，防止过拟合
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout处理，防止过拟合
model.add(Dense(10, activation='softmax'))  # 分类

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(mnist.train.images.reshape((-1, 28, 28, 1)), mnist.train.labels, batch_size=64, epochs=5, verbose=1)

# 保存模型
model.save('model.h5')
```

训练好的模型保存在当前的目录下。
![在这里插入图片描述](https://img-blog.csdnimg.cn/93314e28e2344c1baf4741fe596e71b7.png)


## 二、本地手写数字批量识别 
### 2.1 导入相关依赖
```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
```

### 2.2 数据和模型准备
```python
# 载入数据
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 载入训练好的模型
model = load_model('model.h5')
# 本地图片名称保存在imag_names的列表中
image_names = ['num0.png', 'num1.png', 'num2.png', 'num3.png', 'num4.png',
               'num5.png', 'num6.png', 'num7.png', 'num8.png', 'num9.png']
```
### 2.3 批量预测并显示
```python
# 批量预测本地图片
for image_name in image_names:
    # 本地图片处理
    img = Image.open('./num-draft/' + image_name)  # 导入本地图片
    img_gray = np.array(ImageOps.grayscale(img))  # 图片灰度化
    img_inv = (255 - img_gray) / 255.0  # 转成白底黑字，以适应MNIST数据集中的数据(黑底白字)。再进行归一化处理
    image = img_inv.reshape((1, 28, 28, 1))  # 转四维数据，CNN神经网络预测需要四维数据
    # 预测
    prediction = model.predict(image)  # 预测
    # print(prediction)  # 打印预测结果的数组
    prediction = np.argmax(prediction, axis=1)  # 找出最大值
    print('预测的图片是: ', image_name, ', 预测结果：', prediction)  # 打印预测结果
    # 准备显示
    plt.subplot(2, 5, image_names.index(image_name) + 1)
    plt.imshow(Image.open('./num-draft/' + image_name))
    plt.yticks([])
    plt.title(f'predict:{prediction}', color='blue')
```
### 2.4 完整代码
```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/3/19 16:12
# @Author: AXYZdong
# @File  : dong-Keras-mnist-recognize.py
# @IDE   : Pycharm
# ===============================
# 导入相关依赖库
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

# 载入数据
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 载入训练好的模型
model = load_model('model.h5')
# 本地图片名称保存在imag_names的列表中
image_names = ['num0.png', 'num1.png', 'num2.png', 'num3.png', 'num4.png',
               'num5.png', 'num6.png', 'num7.png', 'num8.png', 'num9.png']
# 批量预测本地图片
for image_name in image_names:
    # 本地图片处理
    img = Image.open('./num-draft/' + image_name)  # 导入本地图片
    img_gray = np.array(ImageOps.grayscale(img))  # 图片灰度化
    img_inv = (255 - img_gray) / 255.0  # 转成白底黑字，以适应MNIST数据集中的数据(黑底白字)。再进行归一化处理
    image = img_inv.reshape((1, 28, 28, 1))  # 转四维数据，CNN神经网络预测需要四维数据
    # 预测
    prediction = model.predict(image)  # 预测
    # print(prediction)  # 打印预测结果的数组
    prediction = np.argmax(prediction, axis=1)  # 找出最大值
    print('预测的图片是: ', image_name, ', 预测结果：', prediction)  # 打印预测结果
    # 准备显示
    plt.subplot(2, 5, image_names.index(image_name) + 1)
    plt.imshow(Image.open('./num-draft/' + image_name))
    plt.yticks([])
    plt.title(f'predict:{prediction}', color='blue')
# 展示图像
plt.text(0, 40, 'By AXYZdong')  # 水印
plt.show()
```
### 2.5 运行结果
![](https://img-blog.csdnimg.cn/411ebb9579ba48088b3f119971fd41d1.png#pic_center)
<p align="center">▲ 窗口输出的打印信息</p>

![在这里插入图片描述](https://img-blog.csdnimg.cn/eb9f458297474af0a02a735350b8bc08.png#pic_center)
<p align="center">▲ 批量识别图像展示</p>

<center><strong>—— END ——</strong></center>



