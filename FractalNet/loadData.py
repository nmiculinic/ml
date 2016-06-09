import os
from six.moves import cPickle as pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

height, width, channels = 32, 32, 3
noClasses = 10

logdir = os.path.expanduser("~/logs/")
test_log_steps = 10
cifar10root = os.path.expanduser('~/Downloads/cifar-10-batches-py')

if not os.path.exists(cifar10root):
    print ("Couldn't find cifar-10 data")
    raise RuntimeError


if os.path.exists(os.path.join(cifar10root, "data.pickle")):
    with open(os.path.join(cifar10root, "data.pickle"), "rb") as f:
        train_data, train_labels, test_data, test_label = pickle.load(f)
else:

    encoder = OneHotEncoder(noClasses, sparse=False)
    encoder.fit(np.arange(10).reshape(-1,1), y=None)

    train_data = []
    train_labels = []

    for i in range(1, 6):
        with open(os.path.join(cifar10root, "data_batch_%d" % i), 'rb') as f:
            data_batch = pickle.load(f, encoding='latin1')
            train_data.append(data_batch['data'])
            train_labels.append(np.array(data_batch['labels']).reshape(-1, 1))

    train_data = np.vstack(train_data)
    train_labels = np.vstack(train_labels)
    train_labels = encoder.transform(train_labels)

    scale = StandardScaler().fit(train_data)
    train_data = scale.transform(train_data).reshape(-1, height, width, channels)

    with open(os.path.join(cifar10root, "test_batch"), 'rb') as f:
        test = pickle.load(f, encoding='latin1')
        test_data = test['data']
        test_label = np.array(test['labels']).reshape(-1, 1)
        test_label = encoder.transform(test_label)

    test_data = scale.transform(test_data).reshape(-1, height, width, channels)

    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_label.shape)

    with open(os.path.join(cifar10root, "data.pickle"), "wb") as f:
        pickle.dump((train_data, train_labels, test_data, test_label), f)
