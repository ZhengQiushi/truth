
----------
# 关于分类器
采用神经网络CNN算法
## 一.文件结构
    ./data
        存放训练集和测试集
    ./Model
        存放训练后的参数信息
    ./tensorflow
	编译好的tensorflow动态库
    ./result
        用于检查测试的分类结果
    ./getNNmodel.ipynb
        训练模型，将参数保存为.pb文件
    ./main.cpp
        测试样例
## 二.使用方法
### 1、总体思路

-->准备编译好的tensorflow动态库

-->准备好训练集

-->运行getNNmodel.ipynb，获得./Model/xxx.pb文件

-->调用分类器函数（三步详见main.cpp）

-->观察result，检查效果

## 三.性能描述

准确率：98%以上

速度：0.7ms per target

## 四.未来优化

1、网络精度优化,可以考虑增大输入的图像大小.或是考虑调整神经网络结构.
	出现的问题,将灯条识别为1

2.是不是可以考虑最后使用softmax而不是logistics,从而进行装甲板打击顺序的选择
