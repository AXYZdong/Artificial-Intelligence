数据分析处理库Pandas，AI常用框架和工具之一。理论知识结合代码实例，希望对您有所帮助。

![在这里插入图片描述](https://img-blog.csdnimg.cn/964bad29e199477198e9c967c2e2edac.png#pic_center)




## 环境说明
>**操作系统：Windows 10** 
>\
> **Python 版本为：Python 3.8.8**	
> \
> **pandas 版本为：1.4.1**

## 一、Pandas简介
Pandas，Python+data+analysis的组合缩写，是python中基于numpy和matplotlib的第三方数据分析库，与后两者共同构成了python数据分析的基础工具包，享有数分三剑客之名。
- **数据格式**：Pandas中包含了高级的数据结构DataFrame和Series。
- **日期处理**：Pandas中包含了时间序列的处理方法，可以生成或者处理日期数据。
- **文件操作**：Pandas可以方便快捷的对CSV，excel和TSV文件进行读写操作。
- **数据分析**：Pandas中提供了大量的方法，用于数据的处理和分析。


## 二、 Pandas中的数据结构
### 2.1 Series
- Series可简单的看作是一维数组。
- Series具有索引（index）。
- Series可以使用字典、数组等数据进行创建。


 **`pandas.Series()` 创建**：pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)：生成一个Series数据。

data数据可以是数组和字典等；index为索引值，要求与数据长度相同，dtype为数据类型。

```python
>>> import numpy as np
>>> import pandas as pd
>>> s = pd.Series([1, 2, 3, np.nan, 5, 6])  #nan缺失值
>>> print(s)
0    1.0
1    2.0
2    3.0
3    NaN
4    5.0
5    6.0
dtype: float64
>>> s.values	#获取数组表现形式
array([ 1.,  2.,  3., nan,  5.,  6.])
>>> s.index		#获取索引对象
RangeIndex(start=0, stop=6, step=1)
```
 **通过 `ndarray` 创建**
```python
>>> import numpy as np
>>> import pandas as pd
>>> a = np.array(['a', 'b', 'c'])
>>> s = pd.Series(a, index = [ 'A', 'B', 'C' ])
>>> print(s)
A    a
B    b
C    c
dtype: object
```
 **通过 `字典` 创建**

```python
>>> import pandas as pd
>>> a = {0:"a", 1:"b", 2:"c"}
>>> s = pd.Series(a)
>>> print(s)
0    a
1    b
2    c
dtype: object
```
### 2.2 DataFrame
DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔值等）。

DataFrame既有行索引，也有列索引，可以看做是Series组成的字典，每个Series看做DataFrame的一个列。

#### 2.2.1 创建DataFrame

- **`pandas.DataFrame()` 创建**

```python
>>> import numpy as np
>>> import pandas as pd
>>> dates = pd.date_range ('20220218',periods = 5)	# 生成作为行索引的5个时间序列
>>> print(dates)
DatetimeIndex(['2022-02-18', '2022-02-19', '2022-02-20', '2022-02-21',
               '2022-02-22'],
              dtype='datetime64[ns]', freq='D')
>>> df = pd.DataFrame (np.random.randn(5,3), index = dates, columns = list("123"))
>>> print(df)
                   1         2         3
2022-02-18 -0.043340  0.296975 -0.129160
2022-02-19  1.004065  0.299931  0.238984
2022-02-20  1.945484 -0.100868 -0.186113
2022-02-21 -0.098250  0.694457 -1.527564
2022-02-22 -2.426660 -0.769743  0.002766
```

**通过 `字典` 创建**

```python
>>> import numpy as np
>>> import pandas as pd
>>> df = pd.DataFrame({'A': 'axyzdong',
                   'B': pd.Timestamp('20220218'),
                   'C': pd.Series(23, index=list(range(3))),
                   'D': np.array([1, 2, 3])})
>>> print(df)
          A          B   C  D
0  axyzdong 2022-02-18  23  1
1  axyzdong 2022-02-18  23  2
2  axyzdong 2022-02-18  23  3
```

#### 2.2.2 查看DataFrame中数据
查看顶部和底部数据、查看索引（行、列）和数据、使用loc方法切片、查看数据详细信息。

```python
import numpy as np
import pandas as pd

# 创建 DateFrame
dates = np.arange(20).reshape(4, 5)
df = pd.DataFrame(dates, index=[1, 2, 3, 4], columns=['a', 'b', 'c', 'd', 'e'])
print(df)
print('--' * 10)

# 查看顶部和底部数据
print(df.head(3))  # 查看前三行数据
print('--' * 10)
print(df.tail(2))  # 查看后两行数据
print('--' * 10)

# 查看索引（行、列）和数据
print("index is :", df.index)  # 输出行索引
print("columns is :", df.columns)  # 输出列索引
print("values is :", df.values)  # 输出数据
print('--' * 10)

# 使用loc方法切片
print(df.loc[1:3:1, 'a']) # 获取a列，索引为1到3（包括3）中的数据，步长为1
print('--' * 10)

# 查看数据详细信息
# count：一列的元素个数；
# mean：一列数据的平均值；
# std：一列数据的均方差；（方差的算术平方根，反映一个数据集的离散程度：越大，数据间的差异越大，数据集中数据的离散程度越高；越小，数据间的大小差异越小，数据集中的数据离散程度越低）
# min：一列数据中的最小值；
# max：一列数中的最大值；
# 25%：一列数据中，前 25% 的数据的平均值；
# 50%：一列数据中，前 50% 的数据的平均值；
# 75%：一列数据中，前 75% 的数据的平均值；）
print(df.describe())
print('--' * 10)
```
运行结果

```python
    a   b   c   d   e
1   0   1   2   3   4
2   5   6   7   8   9
3  10  11  12  13  14
4  15  16  17  18  19
--------------------
    a   b   c   d   e
1   0   1   2   3   4
2   5   6   7   8   9
3  10  11  12  13  14
--------------------
    a   b   c   d   e
3  10  11  12  13  14
4  15  16  17  18  19
--------------------
index is : Int64Index([1, 2, 3, 4], dtype='int64')
columns is : Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
values is : [[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
--------------------
1     0
2     5
3    10
Name: a, dtype: int32
--------------------
               a          b          c          d          e
count   4.000000   4.000000   4.000000   4.000000   4.000000
mean    7.500000   8.500000   9.500000  10.500000  11.500000
std     6.454972   6.454972   6.454972   6.454972   6.454972
min     0.000000   1.000000   2.000000   3.000000   4.000000
25%     3.750000   4.750000   5.750000   6.750000   7.750000
50%     7.500000   8.500000   9.500000  10.500000  11.500000
75%    11.250000  12.250000  13.250000  14.250000  15.250000
max    15.000000  16.000000  17.000000  18.000000  19.000000
--------------------
```
#### 2.2.3 删除DataFrame中数据
```python
import numpy as np
import pandas as pd

# 创建 DateFrame
dates = np.arange(20).reshape(4, 5)
df = pd.DataFrame(dates, index=[1, 2, 3, 4], columns=['a', 'b', 'c', 'd', 'e'])
print(df)

a = df.drop([1], axis=0)    # axis=0时 删除指定的行
b = df.drop(['b'], axis=1)  # axis=1时 删除指定的列
print('------- 删除1行-------')
print(a)
print('------- 删除b列-------')
print(b)
```
运行结果

```python
    a   b   c   d   e
1   0   1   2   3   4
2   5   6   7   8   9
3  10  11  12  13  14
4  15  16  17  18  19
------- 删除1行-------
    a   b   c   d   e
2   5   6   7   8   9
3  10  11  12  13  14
4  15  16  17  18  19
------- 删除b列-------
    a   c   d   e
1   0   2   3   4
2   5   7   8   9
3  10  12  13  14
4  15  17  18  19
```
#### 2.2.4 增加DataFrame中数据

```python
import numpy as np
import pandas as pd

# 创建 DateFrame
dates = np.arange(20).reshape(4, 5)
df = pd.DataFrame(dates, index=[1, 2, 3, 4], columns=['a', 'b', 'c', 'd', 'e'])
print(df)
a = df.head(2) # 查看前两行数据
b = df.drop(['b'], axis=1) # 删除b列
print('------- 数据a-------')
print(a)
print('------- 数据b-------')
print(b)
print('------- 合并-------')
print(a.append(b))
```
运行结果

```python
    a   b   c   d   e
1   0   1   2   3   4
2   5   6   7   8   9
3  10  11  12  13  14
4  15  16  17  18  19
------- 数据a-------
   a  b  c  d  e
1  0  1  2  3  4
2  5  6  7  8  9
------- 数据b-------
    a   c   d   e
1   0   2   3   4
2   5   7   8   9
3  10  12  13  14
4  15  17  18  19
------- 合并-------
    a    b   c   d   e
1   0  1.0   2   3   4
2   5  6.0   7   8   9
1   0  NaN   2   3   4
2   5  NaN   7   8   9
3  10  NaN  12  13  14
4  15  NaN  17  18  19
```

#### 2.2.5 iteritems方法获取数据
`DataFrame.iteritems()`：返回一个由元组组成的可迭代对象。每个元组由DataFrame中列名和所对应的Series组成。

```python
import numpy as np
import pandas as pd

# 创建 DateFrame
dates = np.arange(20).reshape(4, 5)
df = pd.DataFrame(dates, index=[1, 2, 3, 4], columns=['a', 'b', 'c', 'd', 'e'])
print(df)
i = 1
for s in df.iteritems(): #对iteritems产生的元组进行遍历
    print("第%d列数据%s" % (i, s))
    i += 1
```

运行结果
```python
    a   b   c   d   e
1   0   1   2   3   4
2   5   6   7   8   9
3  10  11  12  13  14
4  15  16  17  18  19
第1列数据('a', 1     0
2     5
3    10
4    15
Name: a, dtype: int32)
第2列数据('b', 1     1
2     6
3    11
4    16
Name: b, dtype: int32)
第3列数据('c', 1     2
2     7
3    12
4    17
Name: c, dtype: int32)
第4列数据('d', 1     3
2     8
3    13
4    18
Name: d, dtype: int32)
第5列数据('e', 1     4
2     9
3    14
4    19
Name: e, dtype: int32)
```
## 三、时间序列

时间序列（time series）数据是一种重要的结构化数据形式。在多个时间点观察或测量到的任何时间都可以形成一段时间序列。

- 时间戳：（timestamp）特定的时刻。
- 固定时间：（period）如2022年全年或者某个月份
- 时间间隔：（interval）由起始和结束时间戳表示，时期（period）可以被看做是间隔（interval）的特例。

###  3.1 常用操作
|方法| 说明 |
|--|--|
| pd.DatatimeIndex() | 时间索引 |
| pd.to_datetime() | 时间格式解析 |
| pd.date_range() | 时间戳 |
| pd.period_range() | 生成日期 |
| pd.timedelta_range() | 时间差 |



## 四、数据处理
- 缺失值 NaN：填充、判断、删除
- 统计值：均值、求和、累计等
- 可视化：用绘图的形式来查看数据

### 4.1 缺失数据的简单操作
- **填充**

```python
import numpy as np
import pandas as pd

# 创建 DateFrame
dates = np.arange(12).reshape(3, 4)
df1 = pd.DataFrame(dates, index=[1, 3, 5], columns=['a', 'b', 'c', 'd'])
# 使用reindex方法设置新的索引，多出的索引对应的数据使用NaN填充
df2 = df1.reindex([1, 2, 3, 4, 5])
print(df2)
print('--' * 10)

# 指定数据填充
df3 = df2.fillna(1) # 指定1填充至NaN
print(df3)
```
运行结果

```python
     a    b     c     d
1  0.0  1.0   2.0   3.0
2  NaN  NaN   NaN   NaN
3  4.0  5.0   6.0   7.0
4  NaN  NaN   NaN   NaN
5  8.0  9.0  10.0  11.0
--------------------
     a    b     c     d
1  0.0  1.0   2.0   3.0
2  1.0  1.0   1.0   1.0
3  4.0  5.0   6.0   7.0
4  1.0  1.0   1.0   1.0
5  8.0  9.0  10.0  11.0
```
- **判断**

```python
import numpy as np
import pandas as pd

# 创建 DateFrame
dates = np.arange(12).reshape(3, 4)
df1 = pd.DataFrame(dates, index=[1, 3, 5], columns=['a', 'b', 'c', 'd'])
# 使用reindex方法设置新的索引，多出的索引对应的数据使用NaN填充
df2 = df1.reindex([1, 2, 3, 4, 5])
# isnull方法可以检查数据中是否有空值
print(df2['a'].isnull())    # 检查a列中是否存在空值，是返回True，否返回False
```

运行结果

```python
1    False
2     True
3    False
4     True
5    False
Name: a, dtype: bool
```
- **删除**

```python
import numpy as np
import pandas as pd

# 创建 DateFrame
dates = np.arange(12).reshape(3, 4)
df1 = pd.DataFrame(dates, index=[1, 3, 5], columns=['a', 'b', 'c', 'd'])
# 使用reindex方法设置新的索引，多出的索引对应的数据使用NaN填充
df2 = df1.reindex([1, 2, 3, 4, 5])
print('------- 原始数据-------')
print(df2)
# 删除NaN
df3 = df2.dropna()
print('------- 删除NaN-------')
print(df3)
```

运行结果

```python
------- 原始数据-------
     a    b     c     d
1  0.0  1.0   2.0   3.0
2  NaN  NaN   NaN   NaN
3  4.0  5.0   6.0   7.0
4  NaN  NaN   NaN   NaN
5  8.0  9.0  10.0  11.0
------- 删除NaN-------
     a    b     c     d
1  0.0  1.0   2.0   3.0
3  4.0  5.0   6.0   7.0
5  8.0  9.0  10.0  11.0
```

## 五、读写文本格式的数据
用于将表格型数据读取为DataFrame对象的函数。写入函数类似（`read` 换成 `to`）。

| 函数           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| read_csv       | 从文件、URL、文件型对象中加载带分隔符的数据。默认分隔符为逗号 |
| read_table     | 从文件、URL、文件型对象中加载带分隔符的数据。默认分隔符为制表符（"\t"） |
| read_fwf       | 读取定宽列格式数据（即 无分隔符）                            |
| read_clipboard | 读取剪贴板中的数据，可以看作read_table的剪贴板版。再将网页转换为表格时很有用 |
| read_excel     | 从Excel XLS 或xlsx file 读取表格数据                         |
| read_html      | 读取HTML文档中的说有表格                                     |
| read_json      | 读取JSON字符串中的数据                                       |
| read_pickle    | 读取Python pickle格式中存储的任意对象                        |
| read_sql       | （使用SQLAlchemy）读取SQL查询结果为pandas的DataFrame         |
| read_feather   | 读取Feather二进制文件格式                                    |

**参考文献**
- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about
- [2] https://blog.csdn.net/lemonbit/article/details/106964657


<center><strong>—— END ——</strong></center>



