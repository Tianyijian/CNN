# CNN

 [Convolutional Neural Networks for Sentence Classification (Y.Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)

## 数据集

MELD

## 参数设置

- non-static：使用glove.840B.300d.txt，随训练更新词向量矩阵
- 使用训练、验证和测试集的所有词汇做词表，共13802个，有词向量的6506个（47%）
- dropout=0.5  learning_rate=1.0  max_sent_length=69  class_size=7
- 尽早停止：dev_acc 小于最佳记录三次时，停止训练

## 结果

|      |  train_acc  |   dev_acc   |  test_acc   |
| :--: | :---------: | :---------: | :---------: |
|  1   | 0.756432075 | 0.514878269 | 0.563601533 |
|  2   | 0.714886375 | 0.519386835 | 0.546743295 |
|  3   | 0.715887476 | 0.508566276 | 0.54137931  |
|  4   | 0.804785264 | 0.513976555 | 0.544061303 |
|  5   | 0.729602563 | 0.518485122 | 0.56091954  |
|  6   | 0.828010812 | 0.519386835 | 0.559770115 |
|  7   | 0.838021824 | 0.513976555 | 0.542145594 |
|  8   | 0.759235159 | 0.517583408 | 0.552873563 |
|  9   | 0.698268095 | 0.517583408 | 0.56091954  |
|  10  | 0.709980979 | 0.50405771  | 0.534482759 |
| mean | 0.755511062 | 0.514788097 | 0.550689655 |

## 参考

[Github: galsang/CNN-sentence-classification-pytorch](https://github.com/galsang/CNN-sentence-classification-pytorch)