# 说明           

## 文件说明                                                                     

- data目录下为本次实验涉及的两个csv文件

- img目录下为图片存储

- 根目录下为代码

## 依赖环境说明

1. matplotlib
2. numpy

## 对数几率回归模型建立

定义sigmod函数为
$$
y=\frac{1}{1+e^{-z}}
$$


其中，
$$
z=w^Tx+b
$$
​                            

根据教材定义，通过极大似然估计，得到对率几率回归的损失函数，化简为：

 
$$
L\left( \beta \right) =\sum_{i=1}^m{\left( -y_i\beta ^{\begin{array}{c} T\\\end{array}}\hat{x}_i+\ln \left( 1+e^{\beta ^T\hat{x}_i} \right) \right)}
$$

其中

$$
w^Tx+b=\beta ^T\hat{x}
$$

进一步将其替换为$z$，得到

$$
L=\sum_{i=1}^m{\left( -y_iz_i+\ln \left( 1+e^{z_i} \right) \right)}
$$

预测值$\hat{y}$可以表示为$\frac{1}{1+e^{-z}}$，反解出$z$得到$z=\ln \left( \begin{array}{c} \frac{\hat{y}}{1-\hat{y}}\\\end{array} \right) $，带入上式，可化简为：

$$
L\left( w,b \right) =\sum_{i=1}^m{\left[ -\left( y_i\ln \left( \hat{y}_i \right) +\left( 1-y_i \right) \ln \left( 1-\hat{y}_i \right) \right) \right]}
$$
 
其中$L$是关于参数$w$和$b$的函数，问题转化为求

$$
\left( w^*,b^* \right) =\mathrm{arg}\min _{\left( w,b \right)}L\left( w,b \right)
$$

使用梯度下降法进行求解，如下

$$
\frac{\partial \hat{y}_i}{\partial z}=\hat{y}_i\left( \begin{array}{c} 1-\hat{y}_i\\\end{array} \right) \\\frac{\partial L_i}{\partial z_i}=\frac{\hat{y}_i-y_i}{\hat{y}_i\left( 1-\hat{y}_i \right)}=\hat{y}_i-y_i\\
$$


最终求得有

$$
\frac{\partial L}{\partial w}=\sum_{i=1}^m{\left( \hat{y}_i-y_i \right) x_i}\\\frac{\partial L}{\partial b}=\sum_{i=1}^m{\left( \hat{y}_i-y_i \right)}
$$

由梯度迭代公式：

$$
w_{k+1}=w_k-\alpha \frac{\partial L}{\partial w}\\b_{k+1}=b_k-\alpha \frac{\partial L}{\partial b}
$$
 

## 二分类

可以使用对数几率回归算法进行分类。对输入的数据集，分为数据x和标签y（数据的归类）。首先经过模型fit训练得到参数$w,b$后，线性回归模型$z=w^Tx+b$带入，求得实值$z$，然后利用单位阶跃函数，转化为0-1值，完成二分类。

单位阶跃函数如下：

 $y=\left\{ \begin{array}{c}                                  0, z<0\\ 1\text{，}z\ge 0\\\end{array} \right. $

## 多分类                                           

多分类的实现使用多个二分类器。

这里使用OVR（一对其余）的模式实现，即对于每个分类标签，训练一个以其为正类，其余为负类的二分类器。N分类，需要训练n个2分类器。在测试时，如果只有一个分类器预测为正类，则对应的类别为最终分类结果，如果有多个分类器为正类，则使用置信度最大的类别标记作为分类结果。

简便起见，可以直接使用置信度最大的类别标记作为分类结果，置信度使用sigmod函数输出的值的大小表示。

 

## 数据的预处理

对于数据集的划分，默认采用7：3或者8:2的比例进行训练集/测试集的划分，这里没有划分验证集。数据集的划分尽可能做到均匀和随机，因此，可以选择从输入的数据中，按照标签y分层抽样，对应于程序中split_dataset函数和split_multi_dataset函数（优化后的，通过循环对多个标签分层取样）。



对于二分类，将标签y转化为0-1，便于计算；对于多分类，通过指定本次训练器的正标签，来将其映射为1，其余标签映射为0。


