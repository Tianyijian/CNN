# CNN

 [Convolutional Neural Networks for Sentence Classification (Y.Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)

## 数据集

MELD 文本数据，情绪7分类

## 模型实现

参考：[Github: galsang/CNN-sentence-classification-pytorch](https://github.com/galsang/CNN-sentence-classification-pytorch)

在此基础上新增了以下功能：

- 实现了多次训练求平均，以减小随机误差
- 评价指标由acc扩充至macro_f1、micro_f1、weighted_f1等
- 使用tensorboard可视化loss和train、dev、test数据集上的各种指标
- 清晰记录训练过程中的各项指标变化，并将结果写入文件

## 实验结果

#### 参数设置

- non-static：使用glove.840B.300d.txt，随训练更新词向量矩阵
- **NLTK进行英文分词**，使用训练、验证和测试集的所有词汇做词表，共7687个，有词向量的7125个（92.69%）
- 最大句子长度91，为数据集中最长的句子的长度
- 固定学习率与总epoch数，训练五遍，记录dev_weighted_f1最佳的epoch位置，对五次的test_weighted_f1求平均

|      | learning_rate | epoch |       best epoch       | test_weighted_f1 |
| :--: | :-----------: | :---: | :--------------------: | :--------------: |
|  1   |       1       |  50   |  [32, 36, 39, 21, 23]  |     0.57185      |
|  2   |      0.1      |  200  | [59, 49, 129, 105, 58] |     0.579113     |

## 学习总结

- 英文NLTK分词很重要，而非直接以空格切分，分词前后词汇表为(13802,7687)，有7%结果提升
- macro_f1、micro_f1等指标的原理及实现
- tensorboard可视化很有必要，清晰观察各项指标变化
- pytorch版的CNN简单高效实现

