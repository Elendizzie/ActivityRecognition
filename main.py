
from process_data import *
from cnn import *
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def main():

    #dataset is a phandas data frame
    dataset = read_data(file_path)
    length = len(dataset['x-axis'])


    #normalize each axis
    dataset['x-axis'] = feature_normalize(dataset['x-axis'])
    dataset['y-axis'] = feature_normalize(dataset['y-axis'])
    dataset['z-axis'] = feature_normalize(dataset['z-axis'])

    # for activity in np.unique(dataset["activity"]):
    #     subset = dataset[dataset["activity"] == activity][:180]
    #     plot_activity(activity, subset)

    tmp = []
    for activity in np.unique(dataset["activity"]):
        curset = dataset[dataset["activity"] == activity][:2000]
        tmp.append(curset)

    subset = pd.concat(tmp)


    segment, labels = segment_signal(subset)
    num_segs =  len(segment)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    segment_reshaped = segment.reshape(num_segs, 1, 90, 3)

    print
    print "segment shape: " , segment.shape
    print "labels shape: ", labels.shape
    print "reshaped segment shape: ", segment_reshaped.shape

    #split the dataset into 80% training and 20% testing set
    train_x, train_y, test_x, test_y = data_set_prepare(segment_reshaped, labels)

    create_cnn(train_x, train_y, test_x, test_y)



if __name__ == '__main__':
    main()