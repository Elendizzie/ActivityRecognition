import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import cnn


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


def




def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    print len(data['timestamp'])
    count = 0
    for (start, end) in windows(data['timestamp'], window_size):
        print count
        count += 1
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(data['timestamp'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments, labels


def main():

    raw_data = read_data(file_path)
    print raw_data["x-axis"][0:5]
    print raw_data['timestamp'].shape
    segment_signal(raw_data, window_size=90)


if __name__ == '__main__':
    main()