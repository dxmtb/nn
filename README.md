神经网络大作业
==

介绍
--
本项目是神经网络实验，实现了MLP和CNN。benchmark文件夹保存了测试性能的一些源代码，database用于保存CIFAR-10的数据，需要有Python格式的data_batch_[1-5]和test_batch。nn文件夹保存了供运行的源代码，report文件夹保存了大作业的报告。

如何运行
--
首先在database中放入训练数据和测试数据，并安装程序的依赖包sklearn，python-gflags，scipy，theano，同时要编译程序的C++部分：

```
cd nn
make
```

训练代码如下

```
cd nn
python cifar_mlp.py --hidden_dim 1600 --loss_type mse --activation relu --epoch 10 --lr_W 0.00001 --lr_b 0.00001 #训练测试MLP
python cifar_cnn.py --loss_type softmax --activation relu --epoch 10 #训练测试CNN
```

这样会使用最佳配置训练MLP和CNN，并进行测试。修改epoch可以训练更多的轮数，不过会更慢。
如上命令MLP使用9分钟可以达到50%左右，CNN可能需要1个多小时，建议使用训练好的模型测试。

不训练的话可以使用我训练的模型进行测试。命令如下

```
cd nn
python cifar_mlp.py --hidden_dim 1600 --load_path ../models-best/relu-mse-1600-0.00001-epoch-99.model --loss_type mse --activation relu --epoch 0 #测试MLP
python cifar_cnn.py --load_path ../models-best/new-relu-mse-epoch-9.model --loss_type softmax --activation relu --epoch 0 #测试CNN
```

会输出报告中的最优结果，MLP 52.79%, CNN 70.55%。
