import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
#导入MNIST数据 
from tensorflow.examples.tutorials.mnist import input_data
#全局初始化合适，fan_in是输入节点的数量，fan_out是输出节点的数量，均匀分布的xavier初始化器。
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)
#函数功能：构建函数
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(),
                 scale = 0.1):   #optimizer--优化器，默认为Adam 
        self.n_input = n_input  #输入变量数
        self.n_hidden = n_hidden   #隐含层节点数
        self.transfer = transfer_function   #隐含层激活函数，默认为softplus
        self.scale = tf.placeholder(tf.float32)   #scale--高斯噪声系数，默认为0.1 
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model   R=w2(w1(x+0.1*n)+b1)+b2
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 隐含层
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                self.weights['w1']),
                self.weights['b1']))
        # 输出层
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost   c=（r-x）^2
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        #创建Session对话，初始化类内参数。
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    #参数初始化函数，return:所有初始化的参数。
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
    #执行一步训练，输入样本，一步训练的损失函数 
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X,
                                                                            self.scale: self.training_scale
                                                                            })
        return cost
    #只求cost的损失函数
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,
                                                     self.scale: self.training_scale
                                                     })
    #输出学到的高阶特征 
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,
                                                       self.scale: self.training_scale
                                                       })
    #将隐含层的输出结果作为输入，通过之后的重建测层将提取到的高阶特征复原为原始数据。这个借口和前面的transform
    #正好将整个自编码器拆分为两部分，这里的generate接口是后半部分，将高阶特征复原为原始数据的步骤。
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    #整体运行一遍复原过程，包括提取的高阶特征和通过高阶特征复原数据，即包括transform和generate两块，输入数据是原数据，输出数据是复员以后的数据。
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
                                                               self.scale: self.training_scale
                                                               })
    #获取隐含层的权重W1 
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    #获取隐含层的贬值系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
        
        
        
        
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
#标准化训练、测试数据 
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test
#获取随机block数据，随机抽取样本起始位置。不放回抽样，提高数据的利用率。
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                               n_hidden = 200,
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                               scale = 0.01)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
