# 下载数据集  
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 使用one-hot编码  
print(mnist.train.images.shape, mnist.train.labels.shape)  
print(mnist.test.images.shape, mnist.test.labels.shape)  
print(mnist.validation.images.shape, mnist.validation.labels.shape)  
  
  
import tensorflow as tf  
sess = tf.InteractiveSession()  
# 第一步，定义算法公式  
x = tf.placeholder(tf.float32, [None, 784])  # 构建占位符，None表示样本的数量可以是任意的,784维的向量
W = tf.Variable(tf.zeros([784, 10]))  # 构建一个变量，代表权重矩阵，one hot编码后是10维向量，初始化为0  
b = tf.Variable(tf.zeros([10]))  # 构建一个变量，代表偏置，初始化为0  
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 构建了一个softmax的模型：y = softmax(Wx + b)，y指样本标签的预测值  
  
# 第二步，定义损失函数，选定优化器，并指定优化器优化损失函数  
y_ = tf.placeholder(tf.float32, [None, 10])  
# 交叉熵损失函数  
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  
# 使用梯度下降法最小化cross_entropy损失函数，学习速率为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  
  
# 第三步，迭代地对数据进行训练  
tf.global_variables_initializer().run()
for i in range(1000):  # 迭代次数1000  
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 使用minibatch，一个batch大小为100  
    train_step.run({x: batch_xs, y_: batch_ys})  
  
# 第四步，在测试集或验证集上对准确率进行评测  
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真值  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 用平均值来统计测试准确率  
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))  # 打印测试信息  
