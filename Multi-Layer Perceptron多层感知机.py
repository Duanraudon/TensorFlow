#加载MNIST数据集，创建一个Interactive Session,这样以后执行无需指定Session。
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
#权重初始化为截断的正态分布，其标准差为0.1，
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
#定义x的占位符，Dropout的比率keep_prob即保留节点的概率，作为输入，定义成占位符。
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
#隐含层1，实现Dropout的功能，即随机将一部分节点设置为0，输出层y。
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

#定义损失和优化器，y是预测的概率分布，y'是真实的概率分布，用来判断模型对真实概率分布估计的准确程度。cross_entropy一种损失函数，名为信息熵。
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#全局训练，batch数为100，一共为300000的样本，对全数据集进行了5轮迭代。
tf.global_variables_initializer().run()
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

#测试训练的模型，寻找预测中最大的值，预测数字类别是否是正确的类别，再将布尔值转换为数值，求其平均正确训练值。再加保留节点数为1输出。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
