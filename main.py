from cnn import *
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/actirecog_train', '''
                            Directory where to write event logs and checkpoint.
                            ''')
tf.app.flags.DEFINE_integer('num_gpus', 1, '''
                            Number of GPUs to use
                            ''')

def main():

    train_x = np.load('segmented_data/train_x.npy')
    train_y = np.load('segmented_data/train_y.npy')
    test_x = np.load('segmented_data/test_x.npy')
    test_y = np.load('segmented_data/test_y.npy')

    total_batches = train_x.shape[0]  # total number of segments in training data
    print "total training batches: ", total_batches

    X, Y, loss, SGD_optimizer, accuracy = prepare_cnn(train_x, train_y, test_x, test_y)

    cost_history = np.empty(shape=[1], dtype=float)

    '''
    Start the evaluation session
    '''
    start = time.time()
    print
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf.global_variables_initializer().run()
        for epoch in range(training_epochs):
            for batch in range(total_batches):
                # prepare the data x, y for each training batch
                offset = (batch * batch_size) % (train_y.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :, :, :]
                batch_y = train_y[offset:(offset + batch_size), :]
                _, cost = sess.run([SGD_optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                cost_history = np.append(cost_history, cost)
            print "Epoch: ", epoch, " Training Loss: ", cost, " Training Accuracy: ", \
                sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
        end = time.time()
        print
        print "Total Training Time: ", end-start
        print "Testing Accuracy", sess.run(accuracy, feed_dict={X: test_x, Y: test_y})



if __name__ == '__main__':
    main()