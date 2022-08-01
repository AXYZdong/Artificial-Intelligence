<font color=red><b>可视化库Matplotlib，AI常用框架和工具之一。</b>

![在这里插入图片描述](https://img-blog.csdnimg.cn/0f76b08f7a18437eb902f81568645ce5.png#pic_center)




## 环境说明
>**操作系统：Windows 10** 
> \
> **Python 版本为：Python 3.8.8**	
> \
> **matplotlib 版本为：3.4.3**

## 一、Matplotlib简介

Matplotlib 是 Python 的绘图库。 它可与 NumPy 一起使用，主要用于绘制2D图形。也可绘制3D图形，需要额外安装支持的工具包。

- 在数据分析领域它有很大的地位，而且具有丰富的扩展，能实现更强大的功能。
- 可以生成绘图，包括直方图，功率谱，条形图，错误图，散点图等。
- 文档完善，语法简洁，案例丰富，个性化程度高。


## 二、Matplotlib安装
- Windows下 pip 安装

直接安装

```python
pip install matplotlib
```

或者使用清华的镜像

```python
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Linux 系统也可以使用 Linux 包管理器来安装：

Debian / Ubuntu：

```python
sudo apt-get install python-matplotlib
```

Fedora / Redhat：

```python
sudo yum install python-matplotlib
```


## 三、Matplotlib图像结构

![在这里插入图片描述](https://img-blog.csdnimg.cn/18f2d567324b455ea6ee9f7928facac1.png#pic_center)

<p align="center"> ▲ Matplotlib 图的组成部分 </p>

## 四、Matplotlib基本操作流程

 1. 创建画布：导入工具包，设置画布属性
 2. 绘图：读取数据，根据场景绘制所需图形。设置图形属性：图例、刻度、网格等
 3. 显示：保存图片、显示图片。

**图形属性设置**

|函数| 说明 |
|--|--|
| plt.figure(figsize,dpi) | 画布 |
| plt.lengend() | 图例 |
| plt.xtickets() | 刻度 |
| plt.grid() | 网格 |
| plt.xlabel(),plt.ylabel() | 描述信息 |
| plt.plot(x, y, color=颜色属性, linestyle=‘线的格式', label="") | 颜色和形状 |

## 五、Matplotlib常见图形
- 柱形图

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/2/20 20:29
# @Author: AXYZdong
# @File  : Matplotlib.py
# @IDE   : Pycharm
# ===============================
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
np.random.seed(3)  # 从第三堆种子里面选择随机数（确保每次的随机数一样）
x = 0.5 + np.arange(8)  # 生成0~7的数
y = np.random.uniform(1, 5, len(x))  # 在均匀分布[1,5)中随机采样

# 画图
plt.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

# 显示
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/6eff5af19a4a4648a799bb6a6d0114f2.png#pic_center)
     <p align="center">      ▲ 生成的柱形图       </center>


- 折线图

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # 创建包含单轴的图
ax.plot([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], [1, 1.3, 1.5, 2.2, 2.6, 3.1, 4.5, 5]);  # 在图上画一些点并连线

plt.show() # 显示
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/55c3352753e946ef9a96e85127c8b723.png#pic_center)
     <p align="center">      ▲ 生成的折线图       </center>

- 饼图

```python
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
x = [1, 2, 3, 4, 5]
colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))

# 画图
fig, ax = plt.subplots()
ax.pie(x, colors=colors, radius=3, center=(4, 4),
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)

plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/6ba2f91e98ae44fab8f6ffeb0ec3fcc0.png#pic_center)
     <p align="center">      ▲ 生成的饼状图       </center>


- 散点图

```python
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
np.random.seed(1)  # 从第一堆种子里面选择随机数（确保每次的随机数一样）
x = 4 + np.random.normal(0, 1, 30)
y = 4 + np.random.normal(0, 1, len(x))

sizes = np.random.uniform(20, 80, len(x))  # 在均匀分布[20,80)中随机采样

# plot
fig, ax = plt.subplots()
ax.scatter(x, y, s=sizes, vmin=0, vmax=100)
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/5e8f19446e4a4bd5a848785eb116d1ee.png#pic_center)
     <p align="center">      ▲ 生成的散点图       </center>

## 六、Matplotlib格式化字符

<table class="reference">
<thead>
<tr>
<th>字符</th>
<th>描述</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'-'</code></td>
<td>实线样式</td>
</tr>
<tr>
<td><code>'--'</code></td>
<td>短横线样式</td>
</tr>
<tr>
<td><code>'-.'</code></td>
<td>点划线样式</td>
</tr>
<tr>
<td><code>':'</code></td>
<td>虚线样式</td>
</tr>
<tr>
<td><code>'.'</code></td>
<td>点标记</td>
</tr>
<tr>
<td><code>','</code></td>
<td>像素标记</td>
</tr>
<tr>
<td><code>'o'</code></td>
<td>圆标记</td>
</tr>
<tr>
<td><code>'v'</code></td>
<td>倒三角标记</td>
</tr>
<tr>
<td><code>'^'</code></td>
<td>正三角标记</td>
</tr>
<tr>
<td><code>'&amp;lt;'</code></td>
<td>左三角标记</td>
</tr>
<tr>
<td><code>'&amp;gt;'</code></td>
<td>右三角标记</td>
</tr>
<tr>
<td><code>'1'</code></td>
<td>下箭头标记</td>
</tr>
<tr>
<td><code>'2'</code></td>
<td>上箭头标记</td>
</tr>
<tr>
<td><code>'3'</code></td>
<td>左箭头标记</td>
</tr>
<tr>
<td><code>'4'</code></td>
<td>右箭头标记</td>
</tr>
<tr>
<td><code>'s'</code></td>
<td>正方形标记</td>
</tr>
<tr>
<td><code>'p'</code></td>
<td>五边形标记</td>
</tr>
<tr>
<td><code>'*'</code></td>
<td>星形标记</td>
</tr>
<tr>
<td><code>'h'</code></td>
<td>六边形标记 1</td>
</tr>
<tr>
<td><code>'H'</code></td>
<td>六边形标记 2</td>
</tr>
<tr>
<td><code>'+'</code></td>
<td>加号标记</td>
</tr>
<tr>
<td><code>'x'</code></td>
<td>X 标记</td>
</tr>
<tr>
<td><code>'D'</code></td>
<td>菱形标记</td>
</tr>
<tr>
<td><code>'d'</code></td>
<td>窄菱形标记</td>
</tr>
<tr>
<td><code>'&amp;#124;'</code></td>
<td>竖直线标记</td>
</tr>
<tr>
<td><code>'_'</code></td>
<td>水平线标记</td>
</tr>
</tbody>
</table>

颜色缩写

<table class="reference">
<thead>
<tr>
<th>字符</th>
<th>颜色</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'b'</code></td>
<td>蓝色</td>
</tr>
<tr>
<td><code>'g'</code></td>
<td>绿色</td>
</tr>
<tr>
<td><code>'r'</code></td>
<td>红色</td>
</tr>
<tr>
<td><code>'c'</code></td>
<td>青色</td>
</tr>
<tr>
<td><code>'m'</code></td>
<td>品红色</td>
</tr>
<tr>
<td><code>'y'</code></td>
<td>黄色</td>
</tr>
<tr>
<td><code>'k'</code></td>
<td>黑色</td>
</tr>
<tr>
<td><code>'w'</code></td>
<td>白色</td>
</tr>
</tbody>
</table>


**参考文献**
- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about
- [2] https://www.runoob.com/numpy/numpy-matplotlib.html
- [3] https://matplotlib.org/stable/index.html



     <p align="center"><strong>—— END ——</strong></center>

