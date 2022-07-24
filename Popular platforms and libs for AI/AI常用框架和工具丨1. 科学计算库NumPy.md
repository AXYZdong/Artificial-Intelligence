科学计算库NumPy，AI常用框架和工具之一。理论知识结合代码实例，希望对您有所帮助。
![在这里插入图片描述](https://img-blog.csdnimg.cn/b6f117e138b74e438535f0b55c1d94b4.png)



## 环境说明
>**操作系统：Windows 10** 
>\
> **Python 版本为：Python 3.8.8**	
> \
> **numpy 版本为：1.21.4**

## 一、NumPy简介
NumPy（Numeric Python的缩写），是Python的一个开源的数值计算库。底层是用C语言编写，运行
速度和运行效率远远超过Python代码。

 1. 拥有强大的多维数组以及处理数组的函数集合
 2. 是与C/C++/FORTRAN等代码整合的工具
 3. 具有 科学计算常用的线性代数运算、傅里叶变换、随机数模块等

## 二、数组
### 2.1 NumPy数组的优势
Numpy数组和Python数组两者都可以用于处理多维数组，但相比于Python数组（如列表），NumPy数组有很大的优势。

 1. 减少编程量：可以省略很多循环语句
 2. 增加运算效率：Numpy针对数组进行了优化，存储和输入输出效率都远高于Python数组。
Numpy数组要求元素具有相同的数据类型，避免了类型检查。
 3. 减少内存消耗：Numpy数组占用的内存较少。

### 2.2 ndarray对象
NumPy中核心对象为数组，即ndarray，它描述了相同类型元素的集合。

![在这里插入图片描述](https://img-blog.csdnimg.cn/f1b5e57cdb9d4e81b8bd4bee4cd9b008.png#pic_center)
  <p align="center">▲ ndarray 的内部结构 </p>

- 数组的索引从 0 开始计数；
- 数组中元素类型相同，占用的存储空间相同；
- ndarray包括数据指针、数据类型、维度、跨度。 
### 2.3 创建数组

```python
import numpy as np

a = np.array([1,2,3])   #一维数组

b = np.array([[1,2,3],[4,5,6]]) #二维数组

c = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[2,5,8],[3,6,7],[1,3,5]]]) #三维数组

```
特殊数组创建

```python
import numpy as np

a = np.zeros(3) #全0数组

b = np.ones(3)  #全1数组

c = np.linspace(0, 9, 10)   #等间距数组1

d = np.arange(0, 10, 1) #等间距数组2

print(a)
print(b)
print(c)
print(d)
```
运行结果

```python
[0. 0. 0.]
[1. 1. 1.]
[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
[0 1 2 3 4 5 6 7 8 9]
```

### 2.4 NumPy数据类型
<table class="reference">
<thead>
<tr>
<th>名称</th>
<th>描述</th>
</tr>
</thead>
<tbody>
<tr>
<td>bool_</td>
<td>布尔型数据类型（True 或者 False）</td>
</tr>
<tr>
<td>int_</td>
<td>默认的整数类型（类似于 C 语言中的 long，int32 或 int64）</td>
</tr>
<tr>
<td>intc</td>
<td>与 C 的 int 类型一样，一般是 int32 或 int 64</td>
</tr>
<tr>
<td>intp</td>
<td>用于索引的整数类型（类似于 C 的 ssize_t，一般情况下仍然是 int32 或 int64）</td>
</tr>
<tr>
<td>int8</td>
<td>字节（-128 to 127）</td>
</tr>
<tr>
<td>int16</td>
<td>整数（-32768 to 32767）</td>
</tr>
<tr>
<td>int32</td>
<td>整数（-2147483648 to 2147483647）</td>
</tr>
<tr>
<td>int64</td>
<td>整数（-9223372036854775808 to 9223372036854775807）</td>
</tr>
<tr>
<td>uint8</td>
<td>无符号整数（0 to 255）</td>
</tr>
<tr>
<td>uint16</td>
<td>无符号整数（0 to 65535）</td>
</tr>
<tr>
<td>uint32</td>
<td>无符号整数（0 to 4294967295）</td>
</tr>
<tr>
<td>uint64</td>
<td>无符号整数（0 to 18446744073709551615）</td>
</tr>
<tr>
<td>float_</td>
<td>float64 类型的简写</td>
</tr>
<tr>
<td>float16</td>
<td>半精度浮点数，包括：1 个符号位，5 个指数位，10 个尾数位</td>
</tr>
<tr>
<td>float32</td>
<td>单精度浮点数，包括：1 个符号位，8 个指数位，23 个尾数位</td>
</tr>
<tr>
<td>float64</td>
<td>双精度浮点数，包括：1 个符号位，11 个指数位，52 个尾数位</td>
</tr>
<tr>
<td>complex_</td>
<td>complex128 类型的简写，即 128 位复数</td>
</tr>
<tr>
<td>complex64</td>
<td>复数，表示双 32 位浮点数（实数部分和虚数部分）</td>
</tr>
<tr>
<td>complex128</td>
<td>复数，表示双 64 位浮点数（实数部分和虚数部分）</td>
</tr>
</tbody>
</table>


### 2.5 切片与索引

```python
import numpy as np

a = np.arange(0, 20, 2)
b = a[1]    #索引
c = a[0:5]  #数组切片
d = a[0:8:2]    #步长切片

print(a)
print(b)
print(c)
print(d)
```
运行结果

```python
[ 0  2  4  6  8 10 12 14 16 18]
2
[0 2 4 6 8]
[ 0  4  8 12]
```
- 通过冒号分隔切片参数 **start:stop:step** 来进行切片操作。

```python
import numpy as np

a = np.arange(10)
b = a[0:10:2]	# 从索引 0 开始到索引 10 停止，间隔为 2

print(a)
print(b)
```
运行结果

```python
[0 1 2 3 4 5 6 7 8 9]
[0 2 4 6 8]
```
- 通过布尔运算筛选、过滤得到符合条件的元素。

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a[a > 1])
print(a[a > 2])
```
运行结果

```python
[2 3 4]
[3 4]
```

### 2.6 广播
广播是指NumPy对于不同形状的数组运算的处理能力。具体而言，对于不同形状的数组，对较小的数组进行拓展，使其与较大数组进行匹配。
```python
import numpy as np

a = np.array([1, 1, 1])
b = np.array([[1, 2, 3],[4, 5, 6]])
c = a + b

print(c)
```
运行结果

```python
[[2 3 4]
 [5 6 7]]
```
### 2.7 数组常用方法
|方法| 说明 |
|--|--|
| numpy.reshape | 改变数组的形状、维度 |
| numpy.transpose | 翻转数组（转置） |
| numpy.stack | 连接相同形状的数组 |
| numpy.append | 在数组末尾增加元素 |
| numpy.insert | 在数组指定位置加入元素 |
| numpy.delete | 按照可指定的方式删除数组中的元素，并返回删除后的新数组 |

```python
>>> import numpy as np
>>> a = np.arange(0, 10)   #生成[0,10)之间的数组，步长默认1
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b = np.reshape(a, (2, 5)) #reshape成2行5列
>>> b
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> b.flatten() #默认按行展开
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.transpose(b)	#转置
array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [4, 9]])
>>> np.append(b,[[1,2,3,4,5]],axis=0)	#按照行方向添加元素
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9],
       [1, 2, 3, 4, 5]])
>>> np.insert(b, 1, 6, axis=1)	#按照列方向在指定位置插入元素
array([[0, 6, 1, 2, 3, 4],
       [5, 6, 6, 7, 8, 9]])
>>> c = np.append(b,[[1,2,3,4,5]],axis=0)	#按照行方向添加元素
>>> c
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9],
       [1, 2, 3, 4, 5]])
>>> np.delete(c, 1, 0)	#按照行方向删除指定位置元素
array([[0, 1, 2, 3, 4],
       [1, 2, 3, 4, 5]])
>>> np.stack(([1,2],[3,4]))	#连接相同形状的数组
array([[1, 2],
       [3, 4]])
```

## 三、矩阵
### 3.1 常用方法
NumPy Matrix库（numpy.matlib），即矩阵库，用于进行矩阵运算。
|方法|  说明|
|--|--|
| np.matlib.empty | 返回未初始化的矩阵 |
| np.matlib.zeros | 返回全0矩阵|
| np.matlib.ones| 返回全1矩阵|
| np.matlib.eye | 返回对角矩阵|
| np.matlib.identity| 返回给定大小的单位矩阵|
| np.matlib.rand| 返回指定的随机数填充的矩阵|

```python
>>> import numpy.matlib
>>> import numpy as np
>> np.matlib.empty((2,3))	# 填充为随机数据
matrix([[9.77731297e-312, 9.77731294e-312, 9.77731297e-312],
        [9.77731298e-312, 9.77731298e-312, 9.77731298e-312]])
>>> np.matlib.zeros((2,2))	# 填充0的2X2矩阵
matrix([[0., 0.],
        [0., 0.]])
>>> np.matlib.ones((2,2))	# 填充1的2X2矩阵
matrix([[1., 1.],
        [1., 1.]])
>>> np.matlib.eye(3,4,0,int)	#返回3X4矩阵，对角线索引为0，对角线元素为1，元素int型
matrix([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]])
>>> np.matlib.identity(4,int)	#返回4X4单位阵，元素int型
matrix([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
>>> np.matlib.rand((3,2))	#返回随机的3X2矩阵
matrix([[0.87673631, 0.03012153],
        [0.85038368, 0.57898968],
        [0.52365458, 0.30675083]])
```
### 3.2 数组转化为矩阵
```python
>>> import numpy as np
>>> a = np.array([1,2,3])
>>> np.matrix(a)
matrix([[1, 2, 3]])
>>> np.mat(a)
matrix([[1, 2, 3]])
```
两种方法区别
- 如果输入本身就是一个矩阵，则 `np.mat` 不会对该矩阵make a copy.仅仅是创建了一个新的引用。相当于 `np.matrix(data, copy = False)`。
- 默认为 `np.matrix(data, copy = True)`。创建了一个新的相同的矩阵。当修改新矩阵时，原来的矩阵不会改变。

### 3.3 矩阵运算
- 矩阵相乘
```python
>>> import numpy as np
>>> a = np.mat([[1],[2],[3]])
>>> b = np.mat([1, 2, 3])
>>> a*b
matrix([[1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]])
>>> np.dot(a,b)
matrix([[1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]])
```

- 元素相乘

```python
>>> import numpy as np
>>> a = np.arange(1,5).reshape(2,2)
>>> b = np.array([1,2,3,4]).reshape(2,2)
>>> a
array([[1, 2],
       [3, 4]])
>>> b
array([[1, 2],
       [3, 4]])
>>> np.multiply(a, b)
array([[ 1,  4],
       [ 9, 16]])
>>> a*2
array([[2, 4],
       [6, 8]])
```

- 求逆和转置

```python
>>> import numpy as np
>>> a = np.mat([[1],[2],[3]])
>>> a.I	#求逆
matrix([[0.07142857, 0.14285714, 0.21428571]])
>>> a.T	#求转置
matrix([[1, 2, 3]])
```
## 四、随机模块
### 4.1 NumPy生成随机数
NumPy中随机数模块(random)提供了丰富的随机数相关函数。
|函数| 说明 |
|--|--|
| np.random.rand(d0, d1, ..., dn)  | 输出给定形状，范围在[0,1)的随机数 |
|np.random.randn(d0, d1, ..., dn)|输出在服从标准正态分部的随机数|
|np.random.randint(low[, high, size]) |输出在[low,high)范围的整数|
|np.random.random([size])|输出在[0,1)范围的随机数|

```python
import numpy as np

a = np.random.rand(2, 2)  # rand函数 范围在[0,1)
b = np.random.randn(2, 2)  # randn函数 具有标准正态分布
c = np.random.randint(0, 5)  # randint(low,high,size)函数 返回在[low,high)范围的整数
d = np.random.random((2, 2))  # random函数，和random.rand函数输出相同，输入有区别

print('a=', '\n', a)
print('b=', '\n', b)
print('c=', '\n', c)
print('d=', '\n', d)
```
运行结果（每次运行的结果可能不一样，因为生成的是随机数）

```python
a= 
 [[0.86519472 0.01400421]
 [0.33573726 0.32352532]]
b= 
 [[-1.26230315 -0.45586842]
 [ 0.42427229  0.76411323]]
c= 
 3
d= 
 [[0.37345008 0.20672255]
 [0.62428635 0.6909303 ]]
```

### 4.2 NumPy分布
|函数| 说明 |
|--|--|
| beta(a, b[, size])  | 贝塔分布 |
|lognormal([mean, sigma, size])|对数正态分布|
|normal([loc, scale, size])|正态分布|
|poisson([lam, size])|泊松分布|
|uniform([low, high, size])|均匀分布|

```python
import numpy as np

a = np.random.normal(0, 0.1, 5)  # 正态分布，参数1为均值，参数2为标准差，参数3为返回值的维度
b = np.random.uniform(0, 5, 2)  # 均匀分布
c = np.random.poisson(5, 2)  # 泊松分布

print('a=', '\n', a)
print('b=', '\n', b)
print('c=', '\n', c)
```
运行结果

```python
a= 
 [ 0.10973736  0.09570688 -0.00373881  0.14549571 -0.0531532 ]
b= 
 [1.20295761 1.72379658]
c= 
 [6 4]
```
## 五、常用函数
**算术函数**
  - 三角函数：numpy.sin(), numpy.cos(), numpy.tan()
  - 舍入函数：numpy.around(), numpy.floor(), numpy.ceil()
  - 加减乘除：numpy.add(), numpy.subtract(), mpy.multiply(),numpy.divide() 
  - 余数：numpy.mod()
  - 平方、乘方和平方根：numpy.square(), numpy.pow(), numpy.sqrt()

```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.add(a, b)  # 加
d = np.subtract(a, b)  # 减
e = np.multiply(a, b)  # 乘
f = np.divide(a, b)    # 除
g = np.mod(a, b)       # 取余
h = np.power(a, b)     # 乘方
print('c=', c)
print('d=', d)
print('e=', e)
print('f=', f)
print('g=', g)
print('h=', h)
```
运行结果

```python
c= [5 7 9]
d= [-3 -3 -3]
e= [ 4 10 18]
f= [0.25 0.4  0.5 ]
g= [1 2 3]
h= [  1  32 729]
```

**统计函数**
- 最大最小值：numpy.amin(),numpy.amax()
- 中位数：numpy.median()
- 算术平均：numpy.mean() 
- 加权平均：numpy.average()
- 标准差，方差：numpy.std(),numpy.var()

```python
import numpy as np
a = np.arange(10).reshape(2, 5)
b = np.amin(a, 0)  # 第0维度上最小值
c = np.amax(a, 1)  # 第1维度上最大值
d = np.median(a)   # 中位数
e = np.mean(a)     # 平均数
print('a=', a)
print('b=', b)
print('c=', c)
print('d=', d)
print('e=', e)
```
运行结果

```python
a= [[0 1 2 3 4]
 [5 6 7 8 9]]
b= [0 1 2 3 4]
c= [4 9]
d= 4.5
e= 4.5
```

**参考文献**
- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about
- [2] https://www.runoob.com/numpy/numpy-indexing-and-slicing.html 
- [3] https://blog.csdn.net/qq_43212169/article/details/101679293


<center><strong>—— END ——</strong></center>

