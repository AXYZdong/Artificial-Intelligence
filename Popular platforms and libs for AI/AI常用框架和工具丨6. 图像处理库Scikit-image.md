<font color=red><b>图像处理库Scikit-image，AI常用框架和工具之一。理论知识结合代码实例，希望对您有所帮助。

![在这里插入图片描述](https://img-blog.csdnimg.cn/d9fe9cd0182c4fdb8cf68df211b15f04.png#pic_center)



## 环境说明
>**操作系统：Windows 10** 
> \
> **Python 版本为：Python 3.7.9**	
> \
> **scikit-image	版本为：0.15.0**

## 一、Scikit-image介绍
### 1.1 Scikit-image简介
Scikit-image 是一种开源的用于图像处理的 Python 包。可以应用于以下五个方面：
- 图像分割
- 图像分析
- 色彩操作
- 图像过滤
- 几何变换

### 1.2 Scikit-image模块介绍
|模块| 说明 |
|--|--|
 `io模块`|读取，保存和显示图片和视频。
 `color模块`|图片的颜色空间变换。
 `feature模块`|特征检测和提取，例如，纹理分析等。
 `filters模块`|图像增强，边缘检测，排序滤波器，自动阈值等。
## 二、图像处理
### 2.1 Novice模块
读取图片：`pic = novice.open()`

|方法（其中 pic = novice.open()）| 说明 |
|--|--|
pic.show|展示图片
pic.format|查看图片格式
pic.size|查看图片大小
pic.modified|查看图片是否被修改
pic.compare|图片预览
pic.reset|恢复图片原始状态
pic.save|保存图片

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/3/11 9:00
# @Author: AXYZdong
# @File  : Scikit-image.py
# @IDE   : Pycharm
# ===============================
from skimage import novice
from skimage import data
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = novice.open(data.data_dir + '/coffee.png')
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.imshow(img.array)  # 注意这里要使用img.array数组类型
plt.title('原始图像')
plt.text(600, 450, 'By AXYZdong')  # 水印
a = img.format
w, h = img.size
c = img.modified
d = img.path.endswith('coffee.png')

print('原始图像格式为 %s' % a)
print('原始图像尺寸为 %sx%s' % (w, h))
print('原始图像是否被修改 %s' % c)
print('原始图像名称是否正确 %s ' % d)

img.size = (60, 40)  # 定义图片大小
plt.subplot(1, 2, 2)
plt.imshow(img.array)
plt.title('改变size之后图像')
img.show()
```

运行结果

```python
原始图像格式为 png
原始图像尺寸为 600x400
原始图像是否被修改 False
原始图像名称是否正确 True
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/bbf3e21d8f9c4ec793dac8a2e020ef1b.png#pic_center)
<p align="center">▲ size前后图片对比 </p>


### 2.2 io模块
此模块的一些功能与Novice模块相同。

|方法| 说明 |
|--|--|
Skimage.io.imread(img)|读取图片（读取给定图片，也可以读取skimage中内置图片）
Skimage.io.imshow(arr)|展示图片（展示读取的图片，arr为读取图片的像素矩阵）
Skimage.io.imsave(fname, arr)|保存图片（fname 保存的文件名称，arr 像素矩阵）
Skimage.data.img |内置图片（img为图片名称）

```python
from skimage.io import *
import matplotlib.pyplot as plt
import skimage

img = imread('lena.png')
print(img)  # 查看图片数据
plt.subplot(2, 2, 1)
imshow(img)
moon = skimage.data.moon()
plt.subplot(2, 2, 2)
imshow(moon)
chelsea = skimage.data.chelsea()
plt.subplot(2, 2, 3)
imshow(chelsea)
plt.text(400, -20, 'By AXYZdong', color="blue")  # 水印
checkerboard = skimage.data.checkerboard()
plt.subplot(2, 2, 4)
imshow(checkerboard)
plt.show()
```
运行结果
- 图片数据

```python
[[[223 135 123]
  [224 136 124]
  [222 132 121]
  ...
  [188  97  96]
  [196 108 100]
  [215 127 118]]

 [[222 134 120]
  [223 135 122]
  [221 133 119]
  ...
  [188  97  95]
  [204 112 103]
  [220 127 116]]

 [[221 132 117]
  [224 135 120]
  [222 133 117]
  ...
  [193  99  97]
  [205 109 101]
  [214 117 108]]

 ...

 [[142  62  76]
  [139  59  73]
  [137  58  72]
  ...
  [127  51  79]
  [122  47  77]
  [119  44  75]]

 [[145  67  80]
  [140  62  75]
  [137  59  73]
  ...
  [129  54  82]
  [122  46  77]
  [119  43  74]]

 [[148  70  83]
  [143  65  78]
  [139  61  75]
  ...
  [129  55  82]
  [121  45  76]
  [118  42  72]]]
```
- 本地图片和内置图片展示
![在这里插入图片描述](https://img-blog.csdnimg.cn/2d3843663e504eb8a2b57e3dd786170a.png#pic_center)
<p align="center">▲ 本地图片和内置图片展示 </p>

### 2.3 color模块
此模块主要作用是进行 **颜色空间变换** 。

|方法| 说明 |
|--|--|
convert_colorspace(arr, fromspace, tospace)|将图像数组转换为新的颜色空间<br>fromspace：要转换的颜色空间<br>tospace：转换为的颜色空间
rgb2hsv(rgb)|RGB到HSV颜色空间转换。rgb：rgb格式的图像
hsv2rgb(hsv)|HSV到RGB色彩空间转换。hsv：HSV格式的图像
rgb2gray(rgb)|计算RGB图像的亮度。rgb：rgb格式的图像
separate_stains(rgb, conv_matrix)|RGB染色空间转换
deltaE_cie76(lab1, lab2)|颜色空间中两点之间的欧几里德距离

```python
from skimage.color import *
import skimage
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
img = skimage.data.coffee()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('RGB')
plt.text(600, 500, 'By AXYZdong', color='blue')  # 水印
img_hsv = convert_colorspace(img, 'RGB', 'HSV')   # 或者使用 img_hsv = rgb2hsv(img)
plt.subplot(1, 2, 2)
plt.imshow(img_hsv)
plt.title('HSV')
plt.show()

img_gray = rgb2gray(img)  # 计算RGB图像的亮度
print(img_gray)
```
运行结果

- RGB和HSV图片对比
![在这里插入图片描述](https://img-blog.csdnimg.cn/ea81713bfe8b429798eba02217193916.png#pic_center)
<p align="center">▲ RGB和HSV图片对比 </p>


- RGB图像亮度

```python
[[0.05623333 0.05651608 0.04978902 ... 0.73961804 0.75166549 0.74579451]
 [0.05595059 0.05651608 0.05792275 ... 0.73905255 0.75081725 0.74297412]
 [0.05875608 0.05846549 0.05848824 ... 0.73905255 0.74494627 0.74886784]
 ...
 [0.52715216 0.64400353 0.58801529 ... 0.38923373 0.37580941 0.31533412]
 [0.61655255 0.59387137 0.58352824 ... 0.34640118 0.31699294 0.31195529]
 [0.58801529 0.57484392 0.5757     ... 0.34721176 0.30803373 0.29569569]]
```

### 2.4 filters模块
`Skimage.filters`

- gaussian：多维高斯滤波器。
- sobel：使用Sobel变换查找边缘幅度。
- Prewitt：使用Prewitt变换查找边缘幅度。
- scharr：使用Scharr变换查找边缘幅度。
- median：返回图像的局部中值。
- laplace：使用拉普拉斯算子查找图像的边缘。

**滤镜**

```python
from skimage.filters import *
from skimage import data
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.color import *

img = data.coffee()
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('original')
img_gaussian = gaussian(img, sigma=10, multichannel=True)  # 图片增加高斯滤波
plt.subplot(2, 2, 2)
plt.imshow(img_gaussian)
plt.title('img_gaussian')
img_gray = rgb2gray(img)
plt.subplot(2, 2, 3)
plt.imshow(img_gray, cmap='gray')
plt.text(500, -40, 'By AXYZdong', color='blue')  # 水印
plt.title('img_gray')
med = median(img_gray, disk(5))  # 注意这里图像的格式必须是二维，median是计算二维数组的中位数
plt.subplot(2, 2, 4)
plt.imshow(med)
plt.title('median')
plt.show()
```
运行结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/e30f3502af124900b67ad9f712398774.png#pic_center)
<p align="center">▲ 高斯滤波和局部中值的滤镜图像 </p>
**边缘检测**

```python
from skimage.filters import *
from skimage import data
import matplotlib.pyplot as plt
from skimage.color import *

img = data.coffee()
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('original')
img_gray = rgb2gray(img)  # 首相将图像变为灰度图像（二维）
plt.subplot(2, 2, 2)
plt.imshow(img_gray, cmap="gray")
plt.title('img_gray')
img_edges = sobel(img_gray)  # 使用Sobel变换查找边缘幅度
plt.subplot(2, 2, 3)
plt.imshow(img_edges, cmap="binary")
plt.text(500, -40, 'By AXYZdong', color='blue')  # 水印
plt.title('img_edges')
img_laplace = laplace(data.camera())  # 使用拉普拉斯算子查找图像的边缘（为了显示明显换了一张图片）
plt.subplot(2, 2, 4)
plt.imshow(img_laplace, cmap=plt.cm.gray)
plt.title('img_laplace')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/39a97911b1ec43c0b9c139abd6631c8c.png#pic_center)
<p align="center">▲ Sobel变换和laplace算子边缘检测图像 </p>

## 三、特征提取

`from skimage.feature import *`

- greycomatrix:计算灰度共生矩阵。灰度级共生矩阵是在图像上的给定偏移处共同出现的灰度值的直方图。
- hessian_matrix_eigvals：计算Hessian矩阵的特征值。
- daisy：为给定图像密集地提取DAISY特征描述符。
- canny：边缘使用Canny算法过滤图像。
- hog：提取给定图像的定向梯度直方图（HOG）。

## 四、其他操作

|方法| 说明 |
|--|--|
img_as_float(image, force_copy= False )|将图像转换为浮点格式
img_as_int(image, force_copy= False)|将图像转换为16位有符号整数格式
pad(array, pad_width, mode, ** kwargs)|填充图像
crop(ar, crop_width, copy = False, order =‘K’ )|沿每个维度通过crop_width裁剪阵列ar
invert(image, signed_float= False )|反转图像
measure.compare_psnr(im_true, im_test, data_range=None)|计算图像的峰值信噪比（PSNR）

<br>

**参考文献**

- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about

<center><strong>—— END ——</strong></center>
