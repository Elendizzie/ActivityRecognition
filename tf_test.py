import tensorflow as tf


kernel_size = 60
num_channels = 3
depth = 60

shape = [1, kernel_size, num_channels, depth]
bias_shape = [depth*num_channels]

initial_w = tf.truncated_normal(shape, stddev = 0.1)

initial_c = tf.constant(0.0, shape = bias_shape)
print initial_c

init_wvariable = tf.Variable(initial_w)
init_cvariable = tf.Variable(initial_c)

init_op = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init_op)
    print initial_c.eval()
