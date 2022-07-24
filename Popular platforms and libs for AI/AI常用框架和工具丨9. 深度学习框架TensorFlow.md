<font color=red><b>深度学习框架TensorFlow，AI常用框架和工具之一。理论知识结合代码实例，希望对您有所帮助。

![在这里插入图片描述](https://img-blog.csdnimg.cn/231575b28cd844ba8e428702b650abf4.png)



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
> **注意CUDA、cudnn、Python、tensorflow版本之间的匹配**


## 一、TensorFlow简介
### 1.1 TensorFlow是什么？
TensorFlow 是一个端到端开源机器学习平台。它拥有一个全面而灵活的生态系统，其中包含各种工具、库和社区资源，可助力研究人员推动先进机器学习技术的发展，并使开发者能够轻松地构建和部署由机器学习提供支持的应用。

- 轻松地构建模型：在即刻执行环境中使用 Keras 等直观的高阶 API 轻松地构建和训练机器学习模型，该环境使我们能够快速迭代模型并轻松地调试模型。
- 随时随地进行可靠的机器学习生产：无论您使用哪种语言，都可以在云端、本地、浏览器中或设备上轻松地训练和部署模型。
- 强大的研究实验：一个简单而灵活的架构，可以更快地将新想法从概念转化为代码，然后创建出先进的模型，并最终对外发布。

### 1.2 TensorFlow发展过程

2015年9月0.1   $\implies$  2017年2月1.0  $\implies$  2019年3月2.0

### 1.3 TensorFlow特点 
- **灵活可扩展**：TensorFlow 不是一个严格的“神经网络”库。只要你可以将你的计算表示为一个数据流图，你就可以使用Tensorflow。你来构建图，描写驱动计算的内部循环。我们提供了有用的工具来帮助你组装“子图”(常用于神经网络)，当然用户也可以自己在Tensorflow基础上写自己的“上层库”。定义顺手好用的新复合操作和写一个python函数一样容易，而且也不用担心性能损耗。当然万一你发现找不到想要的底层数据操作，你也可以自己写一点c++代码来丰富底层的操作。
- **多语言**：Tensorflow 有一个合理的c++使用界面，也有一个易用的python使用界面来构建和执行你的graphs。你可以直接写python/c++程序，也可以用交互式的ipython界面来用Tensorflow尝试些想法，它可以帮你将笔记、代码、可视化等有条理地归置好。当然这仅仅是个起点——我们希望能鼓励你创造自己最喜欢的语言界面，比如Go，Java，Lua，Javascript，或者是R。
- **GPU**：支持 GPU。
- **多平台**：Tensorflow 在CPU和GPU上运行，比如说可以运行在台式机、服务器、手机移动设备等等。
- **运算能力强**：由于Tensorflow 给予了线程、队列、异步操作等以最佳的支持，Tensorflow 让你可以将你手边硬件的计算潜能全部发挥出来。你可以自由地将Tensorflow图中的计算元素分配到不同设备上，Tensorflow可以帮你管理好这些不同副本。
- **分布式**：Tensorflow是由高性能的gRPC框架作为底层技术来支持的。这是一个通信框架gRPC(google remote procedure call)，是一个高性能、跨平台的RPC框架。

### 1.4 TensorFlow架构
![在这里插入图片描述](https://img-blog.csdnimg.cn/9493b0a70bd04c70a6c9dd84b5e86c6a.png)
<p align="center">▲ 架构图</p>

## 二、TensorFlow基础知识
### 2.1 TensorFlow组成
![在这里插入图片描述](https://img-blog.csdnimg.cn/cf323e2e3e3d4ea193ccc5de95606774.png#pic_center)
<p align="center">▲ Tensor + Flow </p>

### 2.2 TensorFlow计算过程
![在这里插入图片描述](https://img-blog.csdnimg.cn/9c796bdc95ba40b599829d92d501948e.gif#pic_center =400x)
<p align="center">▲ 计算图实例</p>

### 2.3 TensorFlow基本概念
- **张量(tensor)**：即任意维度的数据，一维、二维、三维、四维等数据统称为张量。
- **算子(operation)**：在TF的实现中，机器学习算法被表达成图，图中的节点是算子(operation)，节点会有0到多个输出。
- **会话(Session)**：客户端使用会话来和TF系统交互，一般的模式是，建立会话，此时会生成一张空图；在会话中添加节点和边，形成一张图，然后执行。
- **变量(Variables)**：机器学习算法都会有参数，而参数的状态是需要保存的。而参数是在图中有其固定的位置的，不能像普通数据那样正常流动。因而，TF中将Variables实现为一个特殊的算子，该算子会返回它所保存的可变tensor的句柄。
- **核(kernel)**：kernel是operation在某种设备上的具体实现。TF的库通过注册机制来定义op和kernel，所以可以通过链接一个其他的库来进行kernel和op的扩展。
- **边(edge)**：正常边，正常边上可以流动数据，即正常边就是tensor。特殊边，又称作控制依赖，(control dependencies)。

## 三、TensorFlow搭建神经网络
### 3.1 TensorFlow开发流程

![在这里插入图片描述](https://img-blog.csdnimg.cn/5c97f3a8fe464f9cac0eb89954612735.png#pic_center =500x)
<p align="center">▲ 开发流程图</p>

### 3.2 TensorFlow神经网络搭建
|网络| 函数方法 |
|--|--|
| 全连接网络 | `tf.layers.dense` |
CNN卷积神经网络|`tf.layers.conv2d`<br>`tf.layers.max_pooling2d`
RNN循环神经网络|`tf.nn.rnn_cell.BasicRNNCell`
其他|`tf.contrib.rnn.BasicLSTMCell`<br>`tf.contrib.rnn.GRUCell`

## 四、代码实例
### 4.1 打印出 Hello tensorflow
```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()      # 建立一个session
print (sess.run(hello))  # 通过session里面的run来运行结果
sess.close()
```
运行结果：

```python
b'Hello, TensorFlow!'
```

### 4.2 基本运算
```python
a = tf.constant(3)  # 定义常量3
b = tf.constant(4)  # 定义常量4

with tf.Session() as sess:  # 建立session
    print("相加: %i" % sess.run(a + b))  # 计算输出两个变量相加的值
    print("相乘: %i" % sess.run(a * b))  # 计算输出两个变量相乘的值
```
运行结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/23d63dec60fb40528f1c464918fba0ef.png)
### 4.3 定义变量

```python
# 定义变量
var1 = tf.Variable(10.0, name="varname")
var2 = tf.Variable(11.0, name="varname")
var3 = tf.Variable(12.0)
var4 = tf.Variable(13.0)
with tf.variable_scope("test1"):
    var5 = tf.get_variable("varname", shape=[2], dtype=tf.float32)

with tf.variable_scope("test2"):
    var6 = tf.get_variable("varname", shape=[2], dtype=tf.float32)

print("var1:", var1.name)
print("var2:", var2.name)
print("var3:", var3.name)
print("var4:", var4.name)
print("var5:", var5.name)
print("var6:", var6.name)
```
运行结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/98479081ea994f4e8ac45d2f06b52dc0.png)





**参考文献**
- [1] https://education.huaweicloud.com/courses/course-v1:HuaweiX+CBUCNXE081+Self-paced/about
- [2] https://tensorflow.google.cn/
- [3] https://www.21ic.com/article/827995.html
- [4] https://blog.csdn.net/DAN_L/article/details/106929907
- [5] https://blog.csdn.net/qq_42635142/article/details/100567418


<center><strong>—— END ——</strong></center>



