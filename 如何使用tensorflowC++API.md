
#  如何使用tensorflowC++API
本篇文章将从一个分类器的实例带你走一遍使用tensorflowC++API搭建网络的流程.
## 1.tensorflow的编译
具体配置过程可以参考:

> https://xugaoxiang.com/2020/05/22/compile-tensorflow2-with-gpu/

我想提醒的是执行编译命令时
```bash
bazel build --verbose-failures --noincompatible_do_not_split_linking_cmdline --config=opt --config=cuda //tensorflow:libtensorflow_cc.so  //tensorflow:install_headers
```
也就是如上语句时
①一开始在fetching阶段,要保持**科#学$上%网^**,否则会获取超时.
②一定是两个,前一个生成动态库,后一个生成头文件,**缺一不可**
③编译完成后,这些文件都是以**软连接**的方式存储在缓存区里的,**建议你将这些库文件拷贝到别的地方**,否则断电就会丢失.~~(重 新 编 译~~ 
```bash
//tensorflow:libtensorflow_cc.so  //tensorflow:install_headers
```
整个过程大概持续1-2小时左右(仅供参考)
![编译好的库文件](https://img-blog.csdnimg.cn/20200707141559950.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MTE0OQ==,size_16,color_FFFFFF,t_70#pic_center)
## 2.Python训练模型并转为pb文件
训练模型存放的就是两部分:网络结构和权重.
接下来我们就介绍如何将keras下的模型转为pb文件

###  搭建网络
```python
import keras
#搭建
happyModel = Model(inputs = X_input, outputs = Y, name='HappyModel')
#编译
happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
#训练
happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)
```
在网络搭建完成后,您可以通过summary命令打印出来模型的网络结构,并记录下输入和输出节点的名称(后续需要使用

```python
happyModel.summary()
```
###  生成pb文件
再一次确认我们的节点名称
```python
#明确输入输出的名字
print('input is :', happyModel.input.name)
print ('output is:', happyModel.output.name)
```
这是我的节点名称:

	input is : input_1:0
	output is: y/Sigmoid:0
将网络转成图的形式,**注意 output_node_names=["y/Sigmoid"] 这里用到了我们模型的节点名称**
```python
#待操作的临时变量sess
sess = K.get_session()
frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    output_node_names=["y/Sigmoid"])
```
将图freeze后保存为pb文件,保存的路径是./param2/param3,这里也就是./Model/happyModel.pb
```python
tf.train.write_graph(frozen_graph_def, 'Model', 'happyModel.pb', as_text=False)
```
可能出现的问题
> #Problem : Cannot find the variable that is input to the ReadVariableOp
   #Solution : 在你的import keras.backend as K 之下添加语句K.set_learning_phase(0)

###  在Python中读取pb文件
来测试一下你的转换是否正确吧.这也可以帮助我们更好理解c++下读取pb的大体流程.恢复之后的网络是tf,而不是keras
我们先把网络恢复出来.注意,这里又一次用到了节点的名字
```python
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
 	#读入pb文件,并转为图
    with open('./Model/happyModel.pb', "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
 	#重新搭建我们的网络,以tf的形式
    with tf.Session() as sess:
        #初始化
        init = tf.global_variables_initializer()
        sess.run(init)
        #明确输入输出
        input_x = sess.graph.get_tensor_by_name("input_1:0")
        output = sess.graph.get_tensor_by_name("y/Sigmoid:0")
```
然后那我们的数据进去跑一跑吧,看看是否结果和预期相符
只需要调用run命令

	参数1:上面定义的输出节点
	参数2:一个数据pair 格式 "名字(string) : 数据(需要的话进行reshape 可以参考之前的summary里边的input)"
```python
print(sess.run(output, feed_dict={input_x: X_ori[3].reshape(1, 32, 32, 1)}))
print(Y_ori[3])
```

## 3.c++下进行predict
好啦,终于到了最后一步.让我们马上开始吧
###  确保动态库可以正常载入
确保两点:
①头文件能够找到
```cpp
#include "tensorflow/core/framework/graph.pb.h"
using tensorflow::Tensor;
using namespace tensorflow;
```
②动态库可以正常工作
这里给出cmake下cmakelist的写法:

```cpp
cmake_minimum_required(VERSION 3.15)
project(myCNN)

# Add Tensorflow headers location to your include paths
include_directories(./tensorflow/include)
include_directories(./tensorflow/include/src)

link_directories(./tensorflow)

# Declare the executable target built from your sources
add_executable(myCNN main.cpp)

# Link your application with Tensorflow libraries
target_link_libraries(myCNN 
./tensorflow/libtensorflow_cc.so ./tensorflowlibtensorflow_framework.so.2)
```
###  载入权重信息
先确定好模型的路径和名字(没错,又是名字
```cpp
/*模型路径*/
const string model_path = "../Model/happyModel.pb";
/*输入输出节点详见ipynb的summary*/
const string input_name = "input_1:0";
const string output_name = "y/Sigmoid:0";
```
然后进行模型的恢复
```cpp
Session* session;
/*创建session*/
Status status = NewSession(SessionOptions(), &session);
GraphDef graph_def;
/*pb文件读入图中*/
status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
/*将模型导入session中*/
status = session->Create(graph_def);
/*设置好我们的input*/
Tensor input(DT_FLOAT, TensorShape({ 1, fixedSize, fixedSize, 1 }));
```
###  测试输出
tf中对输入有特殊的要求,就是要**利用tensor**进行输入.
这里给出opencv下对于测试数据的输入,也就是mat2tensor的过程
```cpp
/*指针指向输入节点*/
float *tensor_data_ptr = input.flat<float>().data();
/*tensor与mat建立起一个映射关系*/
cv::Mat fake_mat(image.rows, image.cols, CV_32FC(image.channels()), tensor_data_ptr);
/*image是你的输入图像,完成tensor的初始化*/
image.convertTo(fake_mat, CV_32FC(image.channels()));
```
输入完成后,就是要计算最后的结果并获取输出.
和输入一样,输出也是保存在最后的tensor中.

> 需要注意的是**session->Run**命令的参数都是vector<xxxx>,这里为了避免非必要的vector声明,我们选择直接用 {
> } 花括号括起.

可以看到输入 **std::pair<string, Tensor>(input_name, input)** 也是一个pair和我们上面的py下的run相对应.
输出我们选择**tensor转scalar**的方法获取
```cpp
/*保留最终输出*/
std::vector<tensorflow::Tensor> outputs;
/*计算最后结果*/
TF_CHECK_OK(session->Run({std::pair<string, Tensor>(input_name, input)}, {output_name}, {}, &outputs));
/*获取输出*/
auto output_c = outputs[0].scalar<float>();
float result = output_c();
```

本文阅读已结束,后续我会在github上分享我的代码以及编译好的tensorflow,需要的欢迎来下载.
