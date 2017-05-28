import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


file_path = "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"

def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    return data

'''
To normailize:
    x' = (x - mean) / sigma
'''
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += (size / 2)

def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    count = 0
    for (start, end) in windows(data['timestamp'], window_size):
        if count % 500 == 0:
            print "Num of segments processed: ", count
        count += 1
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(data['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def data_set_prepare(segment_reshaped, labels):
    train_test_split = np.random.rand(len(segment_reshaped)) < 0.8
    train_x = segment_reshaped[train_test_split]
    train_y = labels[train_test_split]
    test_x = segment_reshaped[~train_test_split]
    test_y = labels[~train_test_split]

    print
    print "train_x shape: ", train_x.shape
    print "train_y shape: ", train_y.shape
    print "test_x shape: ", test_x.shape
    print "test_y shape: ", test_y.shape


    return train_x, train_y, test_x, test_y