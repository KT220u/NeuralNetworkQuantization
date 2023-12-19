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
+ MNIST : 784入力
+ conv1 : 5*5 16枚
+ relu1
+ maxpool1 : 2*2
+ conv2 : 5*5 32枚
+ relu2
+ maxpool2 : 2*2
+ fc1 : 1024ノード全結合層
+ relu3
+ fc2 ; 出力

### 量子化
+ 重みは signed + symmetry
+ 入力、活性値は unsigned
