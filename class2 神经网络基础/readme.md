1. 单个感知器，实现桔子价格预测
网络结构in->node->out
y=ax+b 最简单的单神经元网络，线形激活函数
2. 利用多个感知器实现甜桔子，酸桔子的分类
输入两个特征，中
三层神经网络 in(2)->layer(5 node)->out(2)
激活函数用tanh或者sigmoid都可以或者relu
数据样本生成用scikit的moon 参考：http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

文件说明

data generator.py：用来产生买桔子任务的数据

data generator2.py 用来产生分类桔子任务的数据

nn predict.py：单神经元做线性模型预测

nn classifier.py: 单隐层BP网络做二分类

funcs.py 一些辅助函数

data.txt 预测用数据

data2.txt 分类用数据

课程录像：
链接: https://pan.baidu.com/s/1mieh5Kc 密码: 9s1q
