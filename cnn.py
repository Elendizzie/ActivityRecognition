import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')



def create_cnn(train_x, train_y, test_x, test_y):
    input_height = 1
    input_width = train_x[0][0][0]
    num_labels = train_y[1]
    num_channels = train_x[0][0][1]

    batch_size = 10
    kernel_size = 60
    depth = 60
    hidden_nodes = 1000


    learning_rate = 1e-4
    epochs = 5

    total_batchs = train_x.shape[0] #total number of segments in training data

    X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    c = apply_depthwise_conv(X, kernel_size, num_channels, depth)
    p = apply_max_pool(c, 20, 2)
    c = apply_depthwise_conv(p, 6, depth * num_channels, depth // 10)

    shape = c.get_shape().as_list()
    c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

    f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth // 10), num_hidden])
    f_biases_l1 = bias_variable([hidden_nodes])
    f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))

    out_weights = weight_variable([hidden_nodes, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)