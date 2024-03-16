LR 预测工程

## 快速开始
如何构建、安装、运行
```
//训练
./bin/lr_predict --mode=train  
# 可选参数 
# --algorithm=lr
# --batch=100
# --epoch=30
# --alpha=0.03
# --reg=0.001
# --optimizer=bgd
# --threshold=0.8
# --model_path=./data/lr.model
# --train_path=./data/train.dat
# --test_path=./data/test.dat

//测试
./bin/lr_predict --mode=test

```

## 运行实例
```
B000000301550O:lr-predict chenhanghang$ ./bin/lr_predict --mode=train
Begin train progress..... mode:train train_path:./data/train.dat
epoch 0
auc:0.756632 logLoss:0.206278 acuraccy:0.969945
epoch 1
auc:0.838647 logLoss:0.179379 acuraccy:0.97007
epoch 2
auc:0.861909 logLoss:0.168836 acuraccy:0.970472
epoch 3
auc:0.8731 logLoss:0.162794 acuraccy:0.970748
epoch 4
auc:0.879944 logLoss:0.158913 acuraccy:0.971099
epoch 5
auc:0.884983 logLoss:0.1562 acuraccy:0.971726
epoch 6
auc:0.888327 logLoss:0.154107 acuraccy:0.972178
epoch 7
auc:0.890279 logLoss:0.152358 acuraccy:0.972228
epoch 8
auc:0.897364 logLoss:0.150831 acuraccy:0.972278
epoch 9
auc:0.897603 logLoss:0.149453 acuraccy:0.972428
epoch 10
auc:0.90066 logLoss:0.148176 acuraccy:0.972479
epoch 11
auc:0.899867 logLoss:0.146968 acuraccy:0.972579
epoch 12
auc:0.904657 logLoss:0.145823 acuraccy:0.972629
epoch 13
auc:0.905614 logLoss:0.144731 acuraccy:0.972654
epoch 14
auc:0.904614 logLoss:0.143687 acuraccy:0.97278
epoch 15
auc:0.906331 logLoss:0.142686 acuraccy:0.97288
epoch 16
auc:0.909096 logLoss:0.141726 acuraccy:0.97298
epoch 17
auc:0.911464 logLoss:0.140802 acuraccy:0.973081
epoch 18
auc:0.91182 logLoss:0.139913 acuraccy:0.973281
epoch 19
auc:0.912444 logLoss:0.139057 acuraccy:0.973407
epoch 20
auc:0.913555 logLoss:0.138232 acuraccy:0.973557
epoch 21
auc:0.914732 logLoss:0.137436 acuraccy:0.973557
epoch 22
auc:0.916033 logLoss:0.136668 acuraccy:0.973658
epoch 23
auc:0.916933 logLoss:0.135927 acuraccy:0.973708
epoch 24
auc:0.919167 logLoss:0.135213 acuraccy:0.973884
epoch 25
auc:0.920438 logLoss:0.134523 acuraccy:0.974084
epoch 26
auc:0.920744 logLoss:0.133858 acuraccy:0.97416
epoch 27
auc:0.919715 logLoss:0.133215 acuraccy:0.97421
epoch 28
auc:0.92075 logLoss:0.132595 acuraccy:0.97431
epoch 29
auc:0.922849 logLoss:0.131996 acuraccy:0.974335
```
