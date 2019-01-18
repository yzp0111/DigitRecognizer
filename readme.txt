DigitRecognizer手写数字识别

data文件夹中为原始数据:
....

src中为执行代码：

DigitRecognizer_tf.py文件使用了CNN算法
卷积+relu+池化)*2+展平层+dropout+全连接层+dropout+全连接层
识别准确率为98.8%

RF.py文件使用了RF算法
准确率 97.4%

SVM.py用了SVM算法
准确率92%

KNN.py用了KNN算法
准确率96.6%