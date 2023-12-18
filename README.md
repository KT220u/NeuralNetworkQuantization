# NeuralNetworkQuantization
+ モデルの学習 -> テスト -> 量子化＆テスト
```
$ python3 train.py
$ python3 test.py
$ python3 quantize.py
```
### model1
+ MNIST : 784入力
+ fc1 : 1024ノード全結合層
+ relu
+ fc2 : 出力

### model2

### 量子化
+ 重みは signed + symmetry
+ 入力、活性値は unsigned
