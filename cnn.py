import tensorflow as tf



def kernel_variable(shape):
    '''
    #generate a random normal distribution with given shape
    :param shape: shape of the kernel matrix [height, kernal_length, num_channels, depth]
    :return: init kernel matrix
    '''

    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    generate bias matrix of 0s with given shape
    :param shape: the same shape with filter kernal
    :return: init bias matrix
    '''

    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, K):

    '''
    2D convolution
    :param x: input batch with shape [batch, in_height, in_width, in_channels]
    :param K: input filter kernel with shape [filter_height, filter_width, in_channels, channel_multiplier]
    :param stride: [1,1,1,1] applies the filter to a patch at every offset, stride = 1
    :return: output of original data convolves with the filters
    '''
    conv_output = tf.nn.depthwise_conv2d(x, K, [1, 1, 1, 1], padding='VALID')
    print
    print conv_output
    return conv_output

def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    '''
    generate conv layer
    :param x: input original data
    :param kernel_size: kernel length of filters
    :param num_channels: the width of the filters
    :param depth: the number of filters
    :return: output of conv layer after reLu
    '''

    kernals = kernel_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    conv_output_w_bias = tf.add(depthwise_conv2d(x, kernals), biases)
    print conv_output_w_bias
    return tf.nn.relu(conv_output_w_bias)


def apply_max_pool(x, kernel_size, stride_size):
    '''
    max pooling
    :param x: output of previous layer
    :param kernel_size: size of the pooling layer
    :param stride_size: stride size
    :return: output of the
    '''
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')

def create_cnn(train_x, train_y, test_x, test_y):
    input_height = 1
    input_width = train_x.shape[2]      #length of each chunck, 90
    num_labels = train_y.shape[1]       #length of total num of labels, 6
    num_channels = train_x.shape[3]     # a total of x, y, z, 3 axis


    batch_size = 10
    kernel_size = 60
    depth = 60
    hidden_nodes = 1000


    learning_rate = 1e-4
    training_epochs = 5

    total_batches = train_x.shape[0] #total number of segments in training data
    print "total training batches: ", total_batches

    #shape [None,1,90,3]
    X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])


    conv1 = apply_depthwise_conv(X, kernel_size, num_channels, depth)
    print conv1
    pool1 = apply_max_pool(conv1, 20, 2)
    print pool1
    # c = apply_depthwise_conv(p, 6, depth * num_channels, depth // 10)
    #
    # shape = c.get_shape().as_list()
    # c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])
    #
    # f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth // 10), hidden_nodes])
    # f_biases_l1 = bias_variable([hidden_nodes])
    # f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))
    #
    # out_weights = weight_variable([hidden_nodes, num_labels])
    # out_biases = bias_variable([num_labels])
    # y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)