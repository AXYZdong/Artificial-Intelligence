<font color=red><b>图像处理库Pillow，AI常用框架和工具之一。</b>

![在这里插入图片描述](https://img-blog.csdnimg.cn/b66d02d3f8e247329c4b99b0cc5b6d38.png#pic_center)

## 环境说明
>**操作系统：Windows 10** 
> \
> **Python 版本为：Python 3.8.8**	
> \
> **pillow 版本为：8.4.0**

## 一、Pillow介绍
### 1.1 Pillow简介
Pillow是Python中处理图片的一个工具，功能强大，使用简单，其前身是强大的PIL（Python Image Library）。

- 图像处理：图像基本处理、像素处理、颜色处理等。
- 对图像进行批处理、生成图像预览、图像格式转换等。
- 其他操作：屏幕捕获、图像滤镜等。

### 1.2 Pillow中常见模块
- `ImageDraw`：图像绘图模块，为Image对象提供简单的 2D 图形 。可以使用此模块创建新图像、注释或修饰现有图像，以及动态生成图形以供 Web 使用。
- `ImageFilter`：包含定义为一组预定义的过滤器，可与Image.filter()方法一起使用。
 - `ImageChops`：包含许多算术图像操作，称为通道操作（“chops”）。这些可用于各种目的，包括特殊效果、图像合成、算法绘画等。
- `ImageEnhance`：图像增强模块，包含许多可用于图像增强的类。
- `ImageWin`：支持在 Windows上创建和显示图像。（仅限Windows）
- `ImageOps`：包含许多现成的图像处理操作。这个模块是实验性的，大多数运算符只处理L和RGB图像。

## 二、Pillow安装
- Windows下 pip 安装

直接安装

```python
pip install Pillow
```

或者使用清华的镜像

```python
pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Linux 系统也可以使用 Linux 包管理器来安装：

Debian / Ubuntu：

```python
sudo apt-get install python-dev python-setuptools
```

Fedora / Redhat：

```python
sudo yum install python-devel
```
## 三、Pillow的图像操作

### 3.1 Image
|说明| 函数 |
|--|--|
|图像读取|PIL.Image.open(img)|
|图像处理|PIL.image.eval(image,*args)|
|图像新建|PIL.image.new(mode,size,color=0)|
|图像融合|PIL.Image.merge(mode, bands)|

- `Image.eval`
```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = Image.open('lena.png')  # 获取图像
# img2 = Image.open('bg.JPG')
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("原图像")


def fun(x):
    return x * 0.5


img_eval = Image.eval(img, fun)
plt.subplot(1, 2, 2)
plt.imshow(img_eval)
plt.title("eval处理后的图像")
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/b8d7f40f6e434ba7a39a66893e8242db.png#pic_center)
- `Image.new`

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image
import matplotlib.pyplot as plt

img = Image.new('RGB', (200, 200), 'blue')
plt.imshow(img)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/f3c6fe64dca940619f91f943886f61be.png#pic_center)




-  `Image.merge`

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image
import matplotlib.pyplot as plt

img1 = Image.open('lena.png')  # 获取图像
img2 = Image.open('bg.JPG')

plt.subplot(2, 2, 1)
plt.imshow(img1)
plt.title("img1")

plt.subplot(2, 2, 2)
plt.imshow(img2)
plt.title("img2")

r1, g1, b1 = img1.split()
r2, g2, b2 = img2.split()
img = [r1, b2, g1]

img_merge = Image.merge("RGB", img)
plt.subplot(2, 2, 3)
plt.imshow(img_merge)
plt.text(200, 720, "img_merge")
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/83d3bc23b1ed49c2be50db8b5aa2fbf4.png#pic_center)
### 3.2 图像操作对象
|函数| 说明 |
|--|--|
| img.format | 查看图片的格式 |
img.size| 查看图像的尺寸
img.mode|查看图像的色彩空间
img.thumbnail|图像缩放
img.show|图像展示
img.save|保存图像
img.transpose|图像翻转
img.crop|图像切割
img.rotate|图像逆时针旋转
img.resize|重置图片尺寸

- 查看图片的格式、查看图像的尺寸、查看图像的色彩空间、图像缩放
```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('lena.png')  # 获取图像
plt.subplot(1, 2, 1)
plt.imshow(img)

w, h = img.size
a = img.format
b = img.mode
c = img.info
img.thumbnail((w/2, h/2))
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.xlim(0, 600)
plt.ylim(600, 0)
plt.show()

print('原始图像格式为 %s' % a)
print('原始图像色彩空间为 %s' % b)
print('原始图像尺寸为 %sx%s' % (w, h))
print('原始图像描述信息为 %s ' % c)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/d4bae03636d24db697d49f20dc02158b.png#pic_center)

- 图像翻转、	图像切割、图像逆时针旋转、重置图片尺寸
```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = Image.open('lena.png')  # 获取图像
plt.subplot(3, 2, 1)
plt.imshow(img)
plt.text(650, 310, "原始图像")

img1 = img.transpose(Image.FLIP_TOP_BOTTOM)  # 上下翻转
img2 = img.crop((0, 0, 200, 200))  # 图像切割
img3 = img.rotate(135)  # 图像旋转
img4 = img.resize((200, 200))  # 改变图像尺寸

plt.subplot(3, 2, 3)
plt.imshow(img1)
plt.text(650, 300, "上下翻转后的图像")
plt.subplot(3, 2, 4)
plt.imshow(img2)
plt.text(210, 100, "切割后的图像")
plt.subplot(3, 2, 5)
plt.imshow(img3)
plt.text(650, 300, "旋转后的图像")
plt.subplot(3, 2, 6)
plt.imshow(img4)
plt.text(210, 100, "尺寸改变后的图像")

plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/8763e337e68a4445a37c9858285141ef.png#pic_center)
### 3.3 ImageOps
|函数| 说明 |
|--|--|
ImageOps.autocontrast (image, cutoff= 0, ignore = None )|最大化（标准化）图像对比度。
| ImageOps.crop (image, border = 0 ) | 从所有四个侧面移除相同数量的像素。此功能适用于所有图像模式. |
ImageOps.deform (image, deformer, resample = 2)|变形图像
mageOps.expand (image, border = 0, fill = 0 )|为图像添加边框（填充）
ImageOps.fit (image, size, method=0, bleed=0.0, centering=(0.5, 0.5))|将图像裁剪为指定的尺寸
ImageOps.flip (image)|垂直翻转图像（从上到下）
ImageOps.grayscale (image)|将图像转换为灰度
ImageOps.mirror (image)|水平翻转图像（从左到右）
ImageOps.solarize (image, threshold=128)|翻转高于阈值的所有像素值
ImageOps.invert (image)|翻转颜色通道
ImageOps.posterize (image, bits)|保留Image各通道像素点数值的高bits位

- 最大化图像对比度、剪裁图像、图像填充、图像剪裁为指定尺寸、垂直翻转图像
```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
rom PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = Image.open('lena.png')  # 获取图像
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("原始图像")
# 最大化图像对比度
img1 = ImageOps.autocontrast(img, 10)
plt.subplot(2, 3, 2)
plt.imshow(img1)
plt.title("最大化图像对比度")
# 剪裁图像
img2 = ImageOps.crop(img, (0, 0, 200, 200))
plt.subplot(2, 3, 3)
plt.imshow(img2)
plt.title("剪裁图像")
# 图像填充
img3 = ImageOps.expand(img, (50, 50, 200, 200), 'blue')
plt.subplot(2, 3, 4)
plt.imshow(img3)
plt.title("图像填充")
# 图像剪裁为指定尺寸
img4 = ImageOps.fit(img, (200, 200))
plt.subplot(2, 3, 5)
plt.imshow(img4)
plt.title("图像剪裁为指定尺寸")
plt.text(50, -40, 'By AXYZdong', color="blue")  # 水印
# 垂直翻转图像
img5 = ImageOps.flip(img)
plt.subplot(2, 3, 6)
plt.imshow(img5)
plt.title("垂直翻转图像")
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/f0a27b69e04b41baa25980a923ad81c0.png#pic_center)


- 将图像转换为灰度、水平翻转图像、翻转高于阈值的所有像素值、翻转颜色通道、保留各通道像素点的高bits位

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = Image.open('lena.png')  # 获取图像
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("原始图像")
# 将图像转换为灰度
img1 = ImageOps.grayscale(img)
plt.subplot(2, 3, 2)
plt.imshow(img1, cmap='gray')
plt.title("将图像转换为灰度")
# 水平翻转图像
img2 = ImageOps.mirror(img)
plt.subplot(2, 3, 3)
plt.imshow(img2)
plt.title("水平翻转图像")
# 翻转高于阈值的所有像素值
img3 = ImageOps.solarize(img, 128)
plt.subplot(2, 3, 4)
plt.imshow(img3)
plt.title("反转高于阈值的所有像素值")
# 翻转颜色通道
img4 = ImageOps.invert(img)
plt.subplot(2, 3, 5)
plt.imshow(img4)
plt.title("翻转颜色通道")
plt.text(150, -120, 'By AXYZdong', color="blue")  # 水印
# 保留各通道像素点的高bits位
img5 = ImageOps.posterize(img, 1)
plt.subplot(2, 3, 6)
plt.imshow(img5)
plt.title("保留各通道像素点的高bits位")
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/480870079d21406d9f58d0a34cae7494.png#pic_center)
### 3.4 ImageFilter
|函数| 说明 |
|--|--|
| ImageFilter.BoxBlur (radius) | 通过将每个像素设置为在每个方向上延伸半径像素的方框中的像素的平均值来模糊图像 |
ImageFilter.GaussianBlur ( radius = 2 )|高斯模糊滤镜
ImageFilter.Kernel（size，kernel，scale = None，offset = 0 ）|创建一个卷积核。
ImageFilter.MedianFilter ( size = 3 )|创建中值滤波器。在给定大小的窗口中选取中值像素值
mageFilter.MinFilter ( size = 3 )|创建一个min过滤器。在给定大小的窗口中选取最低像素值。
ImageFilter.MaxFilter ( size = 3 ) |创建一个最大过滤器。选择具有给定大小的窗口中的最大像素值。

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = Image.open('lena.png')  # 获取图像
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("原始图像")
# 模糊图像
img1 = img.filter(ImageFilter.BoxBlur(5))
plt.subplot(2, 3, 2)
plt.imshow(img1)
plt.title("模糊图像")
# 高斯滤波模糊图片
img2 = img.filter(ImageFilter.GaussianBlur(5))
plt.subplot(2, 3, 3)
plt.imshow(img2)
plt.title("高斯滤波模糊图片")
# 轮廓滤波
img3 = img.filter(ImageFilter.CONTOUR)
plt.subplot(2, 3, 4)
plt.imshow(img3)
plt.title("轮廓滤波")
# 中值滤波
img4 = img.filter(ImageFilter.MedianFilter(5))
plt.subplot(2, 3, 5)
plt.imshow(img4)
plt.title("中值滤波")
plt.text(150, -120, 'By AXYZdong', color="blue")  # 水印
# 浮雕滤镜
img5 = img.filter(ImageFilter.EMBOSS)
plt.subplot(2, 3, 6)
plt.imshow(img5)
plt.title("浮雕滤镜")

plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/aa9eadfa6b4a452ca63f77f4a0071e9c.png#pic_center)
### 3.5 ImageEnhance

|函数| 说明 |
|--|--|
| ImageEnhance.Color ( image ) | 调整图像色彩平衡 |
ImageEnhance.Contrast ( image ) |调整图像对比度
ImageEnhance.Brightness ( image ) |调整图像亮度
ImageEnhance.Sharpness ( image ) |调整图像清晰度

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/26 10:34
# @Author: AXYZdong
# @File  : Pillow.py
# @IDE   : Pycharm
# ===============================
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = Image.open('lena.png')  # 获取图像
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title("原始图像")
# 颜色增强
img1 = ImageEnhance.Color(img)
img1 = img1.enhance(2)
plt.subplot(2, 3, 2)
plt.imshow(img1)
plt.title("颜色增强")
# 对比度增强
img2 = ImageEnhance.Contrast(img)
img2 = img2.enhance(2)
plt.subplot(2, 3, 3)
plt.imshow(img2)
plt.title("对比度增强")
# 亮度增强
img3 = ImageEnhance.Brightness(img)
img3 = img3.enhance(2)
plt.subplot(2, 3, 4)
plt.imshow(img3)
plt.title("亮度增强")
# 图像锐化
img4 = ImageEnhance.Sharpness(img)
img4 = img4.enhance(50)
plt.subplot(2, 3, 5)
plt.text(150, -120, 'By AXYZdong', color="blue")  # 水印
plt.imshow(img4)
plt.title("图像锐化")
# 亮度增强
img3 = ImageEnhance.(img)
img3 = img3.enhance(2)
plt.subplot(2, 3, 4)
plt.imshow(img3)
plt.title("亮度增强")

plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/708cb237d1a74c4d827767295b38b48a.png#pic_center)
## 四、绘图功能 
 **ImageDraw**

`ImageDraw.Draw ( im, mode = None )`：画图

`ImageDraw.ellipse ( xy, fill= None, outline= None, width= 0 )`：绘制椭圆

`ImageDraw.line ( xy, fill = None, width = 0, joint = None )`：画直线

`ImageDraw.polygon ( xy, fill = None, outline = None )`：画多边形

```python
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 在创建的画布上画图
blank = Image.new("RGB", [1024, 768], "white")
drawObj = ImageDraw.Draw(blank)
drawObj.line([100, 100, 100, 600], fill='red')  # 直线
drawObj.arc([100, 100, 600, 600], 0, 90, fill='black')  # 画弧线
drawObj.ellipse([100, 100, 300, 300], outline='black', fill='white')  # 画圆
plt.imshow(blank)
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2d210ac6fcd6421590c5ea07aab6e3ef.png#pic_center)

## 五、实例：生成验证码
说明：源代码来源于华为云开发者学堂-AI基础课程常用框架与工具-第四章的pillow实验。

```python
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import random


# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))


# 随机颜色1:
def rndColor():
    return random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)


# 随机颜色2:
def rndColor2():
    return random.randint(32, 127), random.randint(32, 127), random.randint(32, 127)


# 240 x 60:
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Font对象:
font = ImageFont.truetype(font="C:/Windows/Fonts/Arial.ttf", size=36)
# 创建Draw对象:
draw = ImageDraw.Draw(image)
# 填充每个像素:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())
# 输出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())
# 模糊:
image = image.filter(ImageFilter.BLUR)
image.save("验证码.JPG")
image.show()
```
生成的随机验证码如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/fb01e34b95d24a82af7f8242c48f729e.png#pic_center)


**参考文献**

- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about
- [2] https://pillow-cn.readthedocs.io/zh_CN/latest/

<p align="center"><strong>—— END ——</strong></center>

