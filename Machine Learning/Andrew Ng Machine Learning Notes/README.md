# 0. 前言
机器学习是目前信息技术中最激动人心的方向之一。本文以吴恩达老师的机器学习课程为主线，使用 [Process On](https://www.processon.com/i/5ed4fc667d9c08694bedad44) 在线绘图构建机器学习的思维导图。

# 1. 思维导图使用说明
配合吴恩达老师的机器学习视频使用，构建知识脉络，回顾总结复盘。

<font color=red >全部思维导图在线浏览：[吴恩达机器学习丨思维导图丨ProcessOn](https://www.processon.com/view/link/61ee3ca4e401fd06afbaf517)

***需要浏览全图的同学 vx 关注 `AXYZdong` ，回复 `机器学习` 获取密码哦！***

<hr>

# 2. 思维导图主要内容

**引言（Introduction）**

**监督学习部分：**
 1. 单变量线性回归（Linear Regression with One Variable） 
 2. 多变量线性回归（Linear Regression with Multiple Variables） 
 3. 逻辑回归（Logistic Regression）
 4. 正则化（Regularization） 
 5. 神经网络：表述（Neural    Networks:Representation）
 6. 神经网络：学习（Neural Networks:Learning）
 7. 支持向量机（Support Vector Machines）

**无监督学习部分：**
 1. 聚类（Clustering） 
 2. 降维（Dimensionality） 
 3. 异常检测（Anomaly Detection）

**特殊应用：**
 1. 推荐系统（Recommender Systems） 
 2. 大规模机器学习（Large Scale Machine Learning）

**关于建立机器学习系统的建议：**

 1. 应用机器学习的建议（Advice for Applying Machine Learning） 
 2. 机器学习系统的设计（Machine Learning System Design） 
 3. 应用实例：图片文字识别（Application Example: Photo OCR）

<hr>

# 3. 思维导图正文
## 0. 引言（Introduction）
引言部分主要介绍了机器学习的定义、机器学习的相关算法、监督学习和无监督学习。

关于机器学习没有一个统一的定义，下面两条是视频中提到两位学者对机器学习的理解。

- Arthur Samuel (1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.
- Tom Mitchell (1998). Well-posed Learning Problem: A computer program is said to learn from experience E with respect to some task Tand some performance measure P, if its performance on T, as measured by P, improveswith experience E.

![在这里插入图片描述](https://img-blog.csdnimg.cn/cacd405f45d54083b4ed9179097c6cbd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part0 Introduction</code></sup></center><p></p>

## 1. 单变量线性回归（Linear Regression with One Variable）
这部分主要内容包括单变量线性回归的模型表示、代价函数、梯度下降法和使用梯度下降法求解代价函数的最小值。


![在这里插入图片描述](https://img-blog.csdnimg.cn/76ad5cf457d84973bb32500b0e339672.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part1 Linear Regression with One Variable</code></sup></center><p></p>

## 2. 多变量线性回归（Linear Regression with Multiple Variables）
多变量线性回归相当于是单变量的扩展，主要还是按照模型假设、构造代价函数和研究代价函数的最小值这样的思路展开。

与单变量线性回归不同的是，多变量线性回归还可能涉及到特征缩放的问题，主要原因是存在着不同尺度的特征变量，为了使得梯度下降能够快速地收敛，需要将这些特征变量统一尺度（类似于归一化的思想）

相比于单变量线性回归，多变量线性回归在求解代价函数的特征方程时，除了可以使用梯度下降法，还可以使用正则方程。根据特征变量的多少，灵活地选择这两种方法。


![在这里插入图片描述](https://img-blog.csdnimg.cn/141e5a242df24479bb2a0b73a3a21ae3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part2 Linear Regression with Multiple Variables</code></sup></center><p></p>

## 3. 逻辑回归（Logistic Regression）
这里的“回归”不同于线性回归，是一种习惯的叫法。它实质是**分类**，要预测的变量是离散的值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/5c846c14954d46a8885ab8852b63e436.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part3 Logistic Regression</code></sup></center><p></p>

## 4. 正则化（Regularization）
正则化（Regularization）的提出，主要是解决过拟合（over-fitting）的问题。包括线性回归的正则化和逻辑回归的正则化，其本质是通过加入正则化项来保留所有的特征，同时减小参数（特征变量前的系数）的大小。

一个假设在训练数据上能够获得比其他假设更好的拟合， 但是在训练数据外的数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象。出现这种现象的主要原因是训练数据中存在噪音或者训练数据太少。


![在这里插入图片描述](https://img-blog.csdnimg.cn/d21c4ba0019344c9842e9b8c50790e06.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part4 Regularization</code></sup></center><p></p>


## 5. 神经网络：表述（Neural Networks:Representation）
神经网络（Neural Networks）的简要表述，涉及非线性假设、神经网络的模型表示、神经网络的直观理解以及多元分类等内容。

当特征太多时，普通的逻辑回归模型，不能有效地处理这么多的特征，这时候我们需要神经网络。 


![在这里插入图片描述](https://img-blog.csdnimg.cn/3e5ab416add94292a436eb083bfae98d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part5 Neural Networks:Representation</code></sup></center><p></p>


## 6. 神经网络：学习（Neural Networks:Learning）
神经网络（Neural Networks）的代价函数，梯度下降寻求代价函数的最小值，利用反向传播算法（Backpropagation Algorithm）算出梯度下降的方向。

采用梯度的数值检验（Numerical Gradient Checking） 方法，防止代价看上去在不断减小，但最终的结果可能并不是最优解的问题。

如果让初始参数都为0，那么第二层的激活单元将具有相同的值。因此需要初始化参数，采用随机初始化的方法，Python代码如下：

```python
Theta1 = rand(10,11) * (2*eps) - eps
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/a47eca1d16184c0393e2b1c66a29188a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part6 Neural Networks:Learning Learning</code></sup></center><p></p>

## 7. 应用机器学习的建议（Advice for Applying Machine Learning）
- 运用训练好了的模型来预测未知数据的时候发现有较大的误差，下一步应该做什么？运用诊断法判断哪些方法对我们的算法是有效的。

- 利用训练集和测试集评估假设函数是否过拟合，训练集代价函数最小化得到的参数代入测试集代价函数。

- 交叉验证集来帮助选择模型。诊断偏差和方差，算法表现不理想， 要么是偏差比较大，要么是方差比较大。换句话说，出现的情况要么是欠拟合，要么是过拟合问题。

- 学习曲线将训练集误差和交叉验证集误差作为训练集实例数量（m）的函数绘制的图表。


![在这里插入图片描述](https://img-blog.csdnimg.cn/ae320685dac443268a9145a828dee9bf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)</p><center><sup><code>▲ Part7 Advice for Applying Machine Learning</code></sup></center></p>

## 8. 机器学习系统的设计（Machine Learning System Design）
这部分主要内容是误差分析、类偏斜的误差度量 、查准率和查全率之间的权衡和机器学习的数据 。 


![在这里插入图片描述](https://img-blog.csdnimg.cn/01d422240f8844289812d3b2f8af98fb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part8 Machine Learning System Design</code></sup></center><p></p>


## 9. 支持向量机（Support Vector Machines）
支持向量机(Support Vector Machines)实质上是优化逻辑回归中的目标函数，将含有的log项用cost函数代替。

支持向量机用一个最大间距来分离样本，具有鲁棒性，有时被称为大间距分类器。

将核函数(Kernel) 引入支持向量机SVM中，从而代替了对应的高维向量内积。


![在这里插入图片描述](https://img-blog.csdnimg.cn/877dedabf416414d9d7722cea08344e0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part9 Support Vector Machines</code></sup></center><p></p>

## 10. 聚类（Clustering）
聚类（Clustering）无监督学习的一种。

重点算法：K-均值算法。K-Means 是最普及的聚类算法，算法接受一个未标记的数据集，然后将数据聚类成不同的组。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b451914ce0df464cbe8983502d23bdfa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part10 Clustering</code></sup></center><p></p>

## 11. 降维（Dimensionality）
降维（Dimensionality）主要用于数据压缩和数据可视化，也是无监督学习的一种。

重要算法：主成分分析 PCA（Principal Component Analysis）算法。

![在这里插入图片描述](https://img-blog.csdnimg.cn/de4d04ef853a46ddb51f8a890b7c944e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part11 Dimensionality</code></sup></center><p></p>

## 12. 异常检测（Anomaly Detection）

这部分主要包括高斯分布（Gaussian Distribution），使用高斯算法进行异常检测，特征转换将原始数据转换成高斯分布。

重要算法：高斯（Gaussian ）算法。

![在这里插入图片描述](https://img-blog.csdnimg.cn/6c991cc7628a4fd0ba6fe01d4c80a58d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part12 Anomaly Detection</code></sup></center><p></p>

## 13. 推荐系统（Recommender Systems）

这部分内容包括：基于内容的推荐系统、协同过滤 （Collaborative Filtering）、矢量化：低秩矩阵分解 、实施细节：均值归一化 。

重要算法：协同过滤 （Collaborative Filtering）算法。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b801d07e2402464fb23189a92cb8afeb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part13 Recommender Systems</code></sup></center><p></p>

## 14. 大规模机器学习（Large Scale Machine Learning）
主要内容：随机梯度下降法（Stochastic Gradient Descent）、小批量梯度下降（Mini-Batch Gradient Descent）   、随机梯度下降算法的收敛 、在线学习（Online Learning）和 映射简化和数据并行（Map Reduce and Data Parallelism）。   

![在这里插入图片描述](https://img-blog.csdnimg.cn/7b48c58e4bcc49c2b98a73cfbf475c79.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part14 Large Scale Machine Learning</code></sup></center><p></p>

## 15. 应用实例：图片文字识别（Application Example: Photo OCR）
关注**图片文字识别的步骤** 和 <strong>滑动窗口（Sliding Windows）</strong>的使用。
\
![在这里插入图片描述](https://img-blog.csdnimg.cn/7124a0ecd3874484b750b2ca1e5d95cd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part15 Application Example: Photo OCR</code></sup></center><p></p>

## 16. 总结（Conclusion）
![在这里插入图片描述](https://img-blog.csdnimg.cn/0c20927a0fd54f1d956ed9d6272800c4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVhZWmRvbmc=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
</p><center><sup><code>▲ Part16 Conclusion</code></sup></center><p></p>

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
