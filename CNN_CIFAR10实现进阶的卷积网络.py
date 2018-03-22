import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time
#训练轮数，样本尺寸，数据默认路径
max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

#定义初始化权重损失，w1控制L2 loss的大小。L2正则化会让权重不过大。
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

#使用cifar10类下载数据集，解压展开到默认位置
cifar10.maybe_download_and_extract()

#直接调用cifar10_input.distorted_inputs函数
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                            batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)                                                  
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

#定义第一个卷积层，不对第一个卷积层进行L2正则化，进行步长为1的卷积，偏置全为0，用relu进行激活，最大池化3*3尺寸，且步长为2*2，最后正则化

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

#定义第二个卷积层，上一层卷积层输出为64，则第三通道也为64，偏置设为1.0，最后两层调换位置
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')
#定义全连接层，转化向量，获取长度，隐含节点数384
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)
#定义全连接层，隐含节点数为192
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))                                      
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)
#正态分布标准差设为上一层隐含层的倒数，隐含节点为10
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)
#这里把softmax的计算和cross_entropy loss的计算合在了一起，计算均值后，将添加到整体的losses即总的loss，在对loss进行汇总
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
#将logits节点和label_holder传入loss函数获得最终的loss
loss = loss(logits, label_holder)
#选择优化器
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) #0.72
#函数输出结果top k的准确率，默认使用top 1，也就是输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
#全局变量初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#启动前面提到的图片数据增强的线程队列，这里使用16个线程进行加速
tf.train.start_queue_runners()
#在每一个训练过程中，需要先使用session的run方法执行计算，获得一个训练数据，再将其传入，记录每一个step花费的时间，每隔10个step计算并展开当前的loss，
#每秒钟训练的样本数量，以及训练一个batch数据所花费的时间。
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run([images_train,labels_train])
    _, loss_value = sess.run([train_op, loss],feed_dict={image_holder: image_batch, 
                                                         label_holder:label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
    
        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
    
#定义一万个样本，计算模型在top1上的正确样本数，最后求得所有的正确样本的总和。
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0  
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_holder: image_batch,
                                                 label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1
#最后将准确率的评测结果计算并进行输出
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)


