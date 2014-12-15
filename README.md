神经网络大作业
==

介绍
--
本项目是神经网络实验，实现了MLP和CNN。benchmark文件夹保存了测试性能的一些源代码，database用于保存CIFAR-10的数据，需要有Python格式的data_batch_[1-5]和test_batch。nn文件夹保存了供运行的源代码，report文件夹保存了大作业的报告。

如何运行
--
首先在database中放入训练数据和测试数据，并安装程序的依赖包sklearn，python-gflags，scipy，theano。

```
cd nn
python cifar_mlp.py --hidden_dim 1600 --loss_type mse --activation relu --epoch 20 --lr_W 0.001 --lr_b 0.001 #测试MLP
python cifar_cnn.py --loss_type softmax --activation relu --epoch 10 #测试CNN
```

这样会使用最佳配置训练MLP和CNN，并进行测试。修改epoch可以训练更多的轮数，不过会更慢。如上命令MLP可以达到50%左右，CNN可以达到40%。

不训练的话可以使用我训练的模型进行测试，命令如下

```
cd nn
python cifar_mlp.py --load_path --loss_type mse --activation relu --epoch 0 #测试MLP
python cifar_cnn.py --load_path --loss_type softmax --activation relu --epoch 0 #测试CNN
```

会输出报告中的最优结果，MLP %， CNN %。

运行

```
python -m nn.tests.grad_test
```

可以进行梯度测试。