<font color=red><b>深度学习框架Keras，AI常用框架和工具之一。理论知识结合代码实例，希望对您有所帮助。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0523ebc1214f4eec94c86265d555bfa8.png#pic_center)

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

## 一、Keras简介
Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。Keras 最初是作为 ONEIROS 项目（开放式神经电子智能机器人操作系统）研究工作的一部分而开发的。

总的来说，Keras是一个基于 Python 的深度学习库。

Keras适用于：
- 允许简单而快速的原型设计（由于用户友好，高度模块化，可扩展性）。
- 同时支持卷积神经网络和循环神经网络，以及两者的组合。
- 在 CPU 和 GPU 上无缝运行。

**Keras 兼容的 Python 版本: Python 2.7-3.6**

> 为什么取名为 Keras?
> \
> Keras (κέρας) 在希腊语中意为 号角 。它来自古希腊和拉丁文学中的一个文学形象，首先出现于 《奥德赛》 中， 梦神 (Oneiroi, singular Oneiros) 从这两类人中分离出来：那些用虚幻的景象欺骗人类，通过象牙之门抵达地球之人，以及那些宣告未来即将到来，通过号角之门抵达之人。它类似于文字寓意，κέρας (号角) / κραίνω (履行)，以及 ἐλέφας (象牙) / ἐλεφαίρομαι (欺骗)。
> \
> ——Keras中文文档

**Keras特点**

- 用户友好。Keras 提供一致且简单的 API，用户工作量降低，并且在用户错误时提供清晰和可操作的反馈。

- 模块化。 模型是由独立的、完全可配置的模块构成的序列或图。

- 易扩展性。 新的模块很容易添加，现有的模块已经提供了充足的示例。

- 基于 Python 实现。 Keras 没有特定格式的单独配置文件。模型定义在 Python 代码中，这些代码紧凑，易于调试，并且易于扩展。

## 二、Keras模块
| 模块 | 说明 |
|--|--|
keras.layers |用于生成神经网络层，如全连接、RNN、CNN等
keras.model|创建网络模型时所用的API
skeras.optimizers|包含了优化器API，如SGD、Adam等优化器
keras.activations|创建神经网络结构时所需要的激活函数
keras.datasets|Keras中封装好的数据集，如mnist手写数字等学习用的数据集
keras.applications|提供了带有预训练权值的深度学习模型，这些模型可以用来进行预测、特征提取和微调（fine-tuning）
### 2.1 layers
Keras 网络层都有很多共同的函数：

`layer.get_weights()`: 以含有Numpy矩阵的列表形式返回层的权重。

`layer.set_weights(weights)`: 从含有Numpy矩阵的列表中设置层的权重（与get_weights的输出形状相同）。

`layer.get_config()`: 返回包含层配置的字典。

- 全连接层

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```
- 卷积层（以2维数据为例）

2D卷积层（对图像的空间卷积）
```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

- 池化层（以2维数据为例）

对于空间数据的最大池化
```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

- Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

将 Dropout 应用于输入。Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。

### 2.2 models

 1. 模型创建：`Model = Keras.models.Sequential()`
 2. 模型训练/预测：`Model.fit()/Model.predict()`
 3. 保存/加载模型：`Model.save()/keras.models.load()`

### 2.3 optimizers

```python
from keras import optimizers
```

 1.  SGD （随机梯度下降优化器）

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```
参数
- lr: float >= 0. 学习率。
- momentum: float >= 0. 参数，用于加速 SGD 在相关方向上前进，并抑制震荡。
- decay: float >= 0. 每次参数更新后学习率衰减值。
- nesterov: boolean. 是否使用 Nesterov 动量。

2. RMSprop优化器

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```
参数
- lr: float >= 0. 学习率。
- rho: float >= 0. RMSProp梯度平方的移动均值的衰减率.
- epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
- decay: float >= 0. 每次参数更新后学习率衰减值。

3. Adagrad 优化器

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
```
Adagrad 是一种具有特定参数学习率的优化器，它根据参数在训练期间的更新频率进行自适应调整。参数接收的更新越多，更新越小。

参数
- lr: float >= 0. 学习率.
- epsilon: float >= 0. 若为 None, 默认为 K.epsilon().
- decay: float >= 0. 每次参数更新后学习率衰减值.

更多优化器参考：[https://keras.io/zh/optimizers/](https://keras.io/zh/optimizers/)

### 2.4 activations

```python
fromkerasimportactivations
```
激活函数（部分）
- `keras.activations.softmax(x, axis=-1)`：Softmax 激活函数。
- `keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)`：修正线性单元。
- `keras.activations.tanh(x)`：双曲正切激活函数。
- `keras.activations.elu(x, alpha=1.0)`：指数线性单元。

更多激活函数参考：[https://keras.io/zh/activations/](https://keras.io/zh/activations/)

### 2.5 dataset

|常用数据集| 导入 |
|--|--|
CIFAR10|from keras.datasetsimport cifar10
IMDBfrom |keras.datasetsimport imdb
波士顿房屋价格|from keras.datasetsimport boston_housing
newswire话题分类|from keras.datasetsimport reuters
MNIST 手写字符数据集|from keras.datasets import mnist

## 三、Keras创建模型
- 方法一
model.Sequential([layers.Dense(n,input_shape)])

- 方法二
Model=model.Sequential()
Mdel.add(layers.Dense(n,input_shape))

- 方法三
InputShape=Input()
Layer1=layers.Decse(n)(InputShape)

## 四、实例：基于CNN的手写数字识别

### 4.1 模型训练

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


### 4.2 批量识别 

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

![](https://img-blog.csdnimg.cn/411ebb9579ba48088b3f119971fd41d1.png#pic_center)
<p align="center">▲ 窗口输出的打印信息</p>

![在这里插入图片描述](https://img-blog.csdnimg.cn/eb9f458297474af0a02a735350b8bc08.png#pic_center)
<p align="center">▲ 批量识别图像展示</p>


**参考文献**
- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about
- [2] https://keras.io/zh/
- [3] https://blog.csdn.net/great_yzl/article/details/120776341


<center><strong>—— END ——</strong></center>

