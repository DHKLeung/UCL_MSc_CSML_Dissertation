import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_circles, make_classification, make_regression


def train_val_test_split(data, labels, split=(0.6, 0.2, 0.2)):
    # Split data #
    num_data = data.shape[0]
    num_train_data = int(num_data * split[0])
    num_val_data = int(num_data * split[1])
    train_data = data[:num_train_data]
    train_labels = labels[:num_train_data]
    val_data = data[num_train_data:num_train_data + num_val_data]
    val_labels = labels[num_train_data:num_train_data + num_val_data]
    test_data = data[num_train_data + num_val_data:]
    test_labels = labels[num_train_data + num_val_data:]
    train_val_test = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    return train_val_test


def load_skl_data(data_name, need_num=None, split=(0.6, 0.2, 0.2)):
    # Load and unpack data from sklearn & randomise #
    if data_name == 'iris':
        skl_data = load_iris()
    elif data_name == 'wine':
        skl_data = load_wine()
    elif data_name == 'breast_cancer':
        skl_data = load_breast_cancer()
    num_data = skl_data['data'].shape[0]
    random_idx = np.random.permutation(num_data)
    data = skl_data['data'][random_idx]
    labels = skl_data['target'][random_idx]

    # Require number of data #
    if need_num is not None:
        data = data[:need_num]
        labels = data[:need_num]

    # Split data #
    train_val_test = train_val_test_split(data, labels, split=split)
    return train_val_test


def load_circular_data(need_num, noise=0.1, factor=0.5, split=(0.6, 0.2, 0.2)):
    # Load circular data #
    data, labels = make_circles(n_samples=need_num, noise=noise, factor=factor)
    labels[labels == 0] = -1

    # Split data #
    train_val_test = train_val_test_split(data, labels, split=split)
    return train_val_test


def load_two_spirals(need_num, noise=0.5, split=(0.6, 0.2, 0.2)):
    # Create two spirals data #
    n = np.sqrt(np.random.rand(need_num, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(need_num, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(need_num, 1) * noise
    data_extended = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    labels_extended = np.hstack((np.ones(need_num) * -1, np.ones(need_num)))
    idx = np.random.permutation(need_num * 2)
    data_extended = data_extended[idx]
    labels_extended = labels_extended[idx]
    data = data_extended[:need_num]
    labels = labels_extended[:need_num]

    # Split data #
    train_val_test = train_val_test_split(data, labels, split=split)
    return train_val_test


def load_random_classification_dataset(need_num, need_features, need_classes=2, need_flip=0.01, class_sep=1.0, random_state=None, split=(0.6, 0.2, 0.2)):
    # Create data for classification #
    n_informative = need_classes
    n_redundant = 0
    n_repeated = 0
    n_cluster_per_class = 2
    data, labels = make_classification(n_samples=need_num, n_features=need_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=need_classes, n_clusters_per_class=n_cluster_per_class, flip_y=need_flip, class_sep=class_sep, random_state=random_state)

    # Change labels to +1/-1 if it is binary classification #
    if need_classes == 2:
        labels[labels == 0] = -1

    # Split data #
    train_val_test = train_val_test_split(data, labels, split=split)
    return train_val_test


def load_random_regression_dataset(need_num, need_features, bias, noise=1, random_state=None, split=(0.6, 0.2, 0.2)):
    # Create data for regression #
    n_informative = need_features
    n_targets = 1
    data, labels = make_regression(n_samples=need_num, n_features=need_features, n_informative=n_informative, n_targets=n_targets, bias=bias, noise=noise, random_state=random_state)

    # Split data #
    train_val_test = train_val_test_split(data, labels, split=split)
    return train_val_test
