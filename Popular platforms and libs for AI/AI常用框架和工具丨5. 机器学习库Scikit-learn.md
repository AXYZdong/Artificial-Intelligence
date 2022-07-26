<b>机器学习库Scikit-learn，AI常用框架和工具之一。</b>


![在这里插入图片描述](https://img-blog.csdnimg.cn/da6d448b03014029b1b1dfe8132a8fcc.png#pic_center)



## 环境说明
>**操作系统：Windows 10** 
> \
> **Python 版本为：Python 3.8.8**	
> \
> **scikit-learn 版本为：1.0.2**

## 一、Scikit-learn介绍
Scikit-learn(sklearn)是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，包括<strong>回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)</strong>等方法。

Scikit-learn主要包括以下四个部分。

 1. **算法API（Application Programing Interface）**：包含了 **回归**、**分类** 和 **聚类** 等算法。
 2. **特征工程**：包含了 **特征抽取**、**特征预处理** 和 **特征降维** 等
 3. **数据集**：包含一些用于学习的简单数据集，如 **波士顿房价**、**癌症预测** 等。
 4. **模型调优**：模型的一些调优功能，如 **网格搜索**、**k折交叉验证** 等。

## 二、特征工程
### 2.1 特征工程包含的内容
**无量纲化**

- sklearn.preprocessing.MinMaxScaler：归一化

- sklearn.preprocessing.StandardScaler：标准化

**数据编码**

- sklearn.preprocessing. OneHotEncoder：独热编码，适用于离散的特征
- sklearn.preprocessing. LabelEncoder：标签编码，将标签数值化

**特征选择**
- sklearn.feature_selection.VarianceThreshold：低方差特征过滤

其他特征选择算法，包括单变量过滤器选择方法和递归特征消除算法。

| 方法 | 说明 |
|--|--|
`feature_selection.GenericUnivariateSelect` ([…])|具有可配置策略的单变量特征选择器。
`feature_selection.SelectPercentile` ([…])|根据最高分数的百分位数选择特征。
`feature_selection.SelectKBest` ([score_func，k]）|根据k个最高分数选择特征。
`feature_selection.SelectFpr` ([score_func，alpha]）|过滤器：根据FPR测试，在alpha以下选择p值。
`feature_selection.SelectFdr` ([score_func，alpha]）|过滤器：为估计的错误发现率选择p值
`feature_selection.SelectFromModel` (estimator, *)|元转换器，用于根据重要度选择特征。
`feature_selection.SelectFwe` ([score_func，alpha]）|过滤器：选择与Family-wise错误率相对应的p值
`feature_selection.RFE` (estimator, *[, …])|消除递归特征的特征排名。
`feature_selection.RFECV` (estimator, *[, …])|通过消除递归特征和交叉验证最佳特征数选择来进行特征排名。
`feature_selection.VarianceThreshold` ([threshold])|删除所有低方差特征的特征选择器。
`feature_selection.chi2` (X，y)|计算每个非负特征与类之间的卡方统计量。
`feature_selection.f_classif` (X，y)|计算提供的样本的ANOVA F值。
`feature_selection.f_regression` (X，y，* [，中心])|单变量线性回归测试
`feature_selection.mutual_info_classif` (X，y，*)|估计离散目标变量的互信息
`feature_selection.mutual_info_regression` (X，y，*)|估计一个连续目标变量的互信息

**特征降维**
- sklearn.decomposition.PCA：主成分分析 PCA（Principal Component Analysis）算法

### 2.2 特征工程步骤

 **1. 特征提取**

主要是使用转换器类DictVectorizer实例化一个对象，然后在这个对象中调用 `fit_transform` 方法，先拟合数据再将其转化成标准形式。

```python
# =====-*- coding: utf-8 -*-=====
# 本代码参考华为云开发者学堂->AI基础课程->常用框架工具->Python机器学习库Scikit-learn的实验代码
# ===============================
from sklearn.feature_extraction import DictVectorizer

# onehot编码
data = [{'name': 'AXYZdong-1', 'age': 20}, {'name': 'AXYZdong-2', 'age': 24}]
# 转换器类实例化一个对象
transfer = DictVectorizer(sparse=False)
# 调用fit_transform
data = transfer.fit_transform(data)
print("返回的结果:\n", data)
print("特征名字：\n", transfer.get_feature_names_out())
```
运行结果

```python
返回的结果:
 [[20.  1.  0.]
 [24.  0.  1.]]
特征名字：
 ['age' 'name=AXYZdong-1' 'name=AXYZdong-2']
```

 **2. 特征预处理**

- 归一化

**适用范围** ：最大值与最小值非常容易受异常点影响，所以归一化法鲁棒性较差，只适合传统精确小数据场景。

```python
import pandas as pd  # 导入数据分析处理库 pandas
from sklearn.preprocessing import MinMaxScaler  # 归一化API

# 创建原始数据
a = [20, 30, 40]
b = [12, 13, 14]
c = [32, 33, 34]

df = pd.DataFrame([a, b, c], columns=['a', 'b', 'c'])  # 创建 DateFrame
transfer = MinMaxScaler(feature_range=(0, 1))  # 实例化一个转换器对象
date = transfer.fit_transform(df[['a', 'b', 'c']])  # 调用fit_transform进行转化
print(date)  # 打印转化结果
```
运行结果

```python
[[0.4        0.85       1.        ]
 [0.         0.         0.        ]
 [1.         1.         0.76923077]]
```

- 标准化

**适用范围**：异常值对我影响小，适合现代嘈杂大数据场景 **(经常使用)**

```python
import pandas as pd  # 导入数据分析处理库 pandas
from sklearn.preprocessing import StandardScaler  # 标准化API

# 创建原始数据
a = [20, 30, 40]
b = [12, 13, 14]
c = [32, 33, 34]

df = pd.DataFrame([a, b, c], columns=['a', 'b', 'c'])  # 创建 DateFrame
transfer = StandardScaler() # 实例化一个标准化对象
date = transfer.fit_transform(df[['a', 'b', 'c']])  # 调用fit_transform进行转化
print(date)  # 打印转化结果
```
运行结果

```python
[[-0.16222142  0.52990781  0.95961623]
 [-1.13554995 -1.40047065 -1.37944833]
 [ 1.29777137  0.87056284  0.4198321 ]]
```
- 两者区别

归一化是将样本的特征值转换到同一量纲下，把数据映射到 [0,1] 或者 [-1,1] 区间内，仅由变量的极值决定，因区间放缩法是归一化的一种。

标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，转换为标准正态分布，和整体样本分布相关，每个样本点都能对标准化产生影响。

它们的相同点在于都能取消由于量纲不同引起的误差；都是一种线性变换，都是对向量X按照比例压缩再进行平移。

## 三、回归算法
### 3.1 线性回归API
- **线性回归**
`sklearn.linear_model.LinearRegression(fit_intercept=True)`  
`sklearn.linear_model.SGDRegressor(loss, fit_intercept=True, learning_rate)`

### 3.2 线性回归实例
实例：利用线性回归预测波士顿房价

```python
# =====-*- coding: utf-8 -*-=====
# @Time  : 2022/3/1 8:29
# @Author: AXYZdong
# @IDE   : Pycharm
# ===============================
from sklearn.datasets import load_boston  # 导入数据——波士顿房价
from sklearn.linear_model import SGDRegressor  # 线性回归的SDG算法（随机梯度下降算法）
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.preprocessing import StandardScaler  # 数据标准化
import matplotlib.pyplot as plt  # 导入可视化库

boston = load_boston()  # 加载数据集
# 将数据集划分成测试集和训练集
train_data, test_data, train_result, test_result = train_test_split(boston.data, boston.target, test_size=0.1, random_state=1)
# 数据标准化
transfer = StandardScaler()  # 实例化一个标准化对象
train_data = transfer.fit_transform(train_data)  # 调用fit_transform进行转化
test_data = transfer.fit_transform(test_data)
# 线性回归算的实现
estimator = SGDRegressor()  # 实例化一个线性回归对象
estimator.fit(train_data, train_result)  # 填充数据进行训练
# 预测
predict_result = estimator.predict(test_data)
# 查看可视化结果
plt.figure(figsize=(10, 8))
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.plot([i for i in range(len(test_data))], test_result, linestyle=':', marker='^', label='true')
plt.plot([i for i in range(len(test_data))], predict_result, linestyle=':', marker='o', label='predict')
plt.legend()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/5172b53fcec44a698f8d55c9e10e2a11.png#pic_center)
  <p align="center">▲ 可视化结果</p>

## 四、分类算法
### 4.1 分类算法API
- **逻辑回归**：`sklearn.linear_model.LogisticRegression`
- **决策树**：`sklearn.tree.DecisionTreeClassifier(criterion,max_depth=None,random_state=None)`
- **KNN**：`sklearn.neighbors.KNeighborsClassifier(n_neighbors,algorithm='auto')`
- **朴素贝叶斯**：`sklearn.naive_bayes.MultinomialNB(alpha = 1.0)`

### 4.2 分类算法实例
实例：使用KNN算法实现鸢尾花种类预测。

```python
# =====-*- coding: utf-8 -*-=====
# 本代码参考华为云开发者学堂->AI基础课程->常用框架工具->Python机器学习库Scikit-learn的实验代码
# ===============================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
# x_train,x_test,y_train,y_test为训练集特征值、测试集特征值、训练集目标值、测试集目标值
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=6)  # random_state 随机种子，确保每次随机的数据相同
transfer = StandardScaler()  # 实例化一个标准化对象
x_train = transfer.fit_transform(x_train)  # 调用fit_transform进行转化
x_test = transfer.fit_transform(x_test)
# 实例化KNN分类器
estimator = KNeighborsClassifier(n_neighbors=10)
estimator.fit(x_train, y_train)
# 模型评估
y_predict = estimator.predict(x_test)
print("预测结果为:\n", y_predict)
print("比对真实值和预测值：\n", y_predict == y_test)
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)
```

运行结果：

```python
预测结果为:
 [0 2 0 0 2 1 2 0 2 1 1 1 2 2 1 1 2 1 1 0 0 2 0 0 1 1 1 2 0 1]
比对真实值和预测值：
 [ True  True  True  True  True  True  True  True  True  True False  True
  True  True  True False  True  True  True  True  True  True  True  True
  True  True  True  True  True  True]
准确率为：
 0.9333333333333333
```

## 五、聚类算法
### 5.1 聚类算法API
- **K-means**：基于欧式距离的聚类算法。
`sklearn.cluster.KMeans(n_clusters=8,init=‘k-means++’)`

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**：基于密度的噪声应用空间聚类，是一个比较有代表性的基于密度的聚类算法。
`sklearn.cluster.DBSCAN`

### 5.2 聚类算法实例

实例：使用 K-means 聚类算法实现鸢尾花的聚类操作。

```python
# 导入依赖库
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
# 数据加载与选择特征
iris = load_iris()
X = iris.data[:, :4]  # #表示我们取特征空间中的4个维度
# 绘制数据分布图
img0 = plt.subplot(1, 2, 1)
plt.scatter(X[:, 2], X[:, 3], c="blue", marker='^', label='see')
plt.text(6.5, -0.3, 'petal length')
plt.ylabel('petal width')
plt.title('original')
plt.legend(loc=2)
# 聚类阶段
estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
img1 = plt.subplot(1, 2, 2)
plt.scatter(x0[:, 2], x0[:, 3], c="red", marker='o', label='label0')
plt.scatter(x1[:, 2], x1[:, 3], c="green", marker='*', label='label1')
plt.title('classification')
plt.legend(loc=2)
# 展示最终效果
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/6ada6a8993ed44589812424563211406.png#pic_center)
  <p align="center">▲ 原始数据与聚类之后对比 </p>


## 六、其他操作
**模型调优**

- K折交叉验证

为算法设置CV参数，即指定交叉验证的K 
```python
sklearn.model_selection.KFold
```

- 网格搜索(GridSearch)
```python
sklearn.model_selection.GridSearchCV(estimator,param_grid=None,cv=None)
```

- 随机搜索
 ```python
sklearn.model_selection:RandomizedSearchCV
 ```

**模型保存和加载**

- 模型保存
 ```python
from sklearn.externals import joblibjoblib.dump(model, 'model.pkl')
 ```

- 模型加载
 ```python
from sklearn.externals import joblibestimator = joblib.load('model.pkl')
 ```

<br>

**参考文献**

- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about
- [2] https://scikit-learn.org.cn/
- [3] https://blog.csdn.net/lian740930980/article/details/114124167

<p align="center"><strong>—— END ——</strong></center>
