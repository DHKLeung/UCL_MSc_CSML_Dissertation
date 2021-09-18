import numpy as np


def lmbda_vec_beta(alpha):
    lmbda = np.random.beta(alpha[0], alpha[1])
    lmbda_vec = np.array([lmbda, 1 - lmbda])
    return lmbda_vec


def cvxcomb_mixup(data, labels, lmbda_vec_func):
    idxs = np.random.choice(data.shape[0], 2, replace=False)
    two_samples = data[idxs]
    two_labels = labels[idxs]
    lmbda_vec = lmbda_vec_func().reshape(-1, 1)
    synthesis_data = np.dot(lmbda_vec.T, two_samples)
    synthesis_label = np.squeeze(np.dot(lmbda_vec.T, two_labels.reshape(-1, 1)))
    return synthesis_data, synthesis_label


def cvxcomb_mixup_sameclass(data, labels, lmbda_vec_func):
    unique_class = np.unique(labels)
    target_class = np.random.choice(unique_class)
    data_target_class = data[labels == target_class]
    idxs = np.random.choice(data_target_class.shape[0], 2, replace=False)
    two_samples = data_target_class[idxs]
    lmbda_vec = lmbda_vec_func().reshape(-1, 1)
    synthesis_data = np.dot(lmbda_vec.T, two_samples)
    return synthesis_data, target_class


def cvxcomb_mixup_nearestneighbour(data, labels, lmbda_vec_func):
    idx = np.random.choice(data.shape[0])
    sample_1 = data[idx].reshape(1, -1)
    label_1 = labels[idx]
    dists = np.argsort(np.linalg.norm(sample_1 - data, axis=1))
    sample_2 = data[dists[1]].reshape(1, -1)
    label_2 = labels[dists[1]]
    two_samples = np.vstack((sample_1, sample_2))
    two_labels = np.hstack((label_1, label_2))
    lmbda_vec = lmbda_vec_func().reshape(-1, 1)
    synthesis_data = np.dot(lmbda_vec.T, two_samples)
    synthesis_label = np.squeeze(np.dot(lmbda_vec.T, two_labels.reshape(-1, 1)))
    return synthesis_data, synthesis_label


def cvxcomb_mixup_sameclass_nearestneighbour(data, labels, lmbda_vec_func):
    unique_class = np.unique(labels)
    target_class = np.random.choice(unique_class)
    data_target_class = data[labels == target_class]
    idx = np.random.choice(data_target_class.shape[0])
    sample_1 = data_target_class[idx].reshape(1, -1)
    dists = np.argsort(np.linalg.norm(sample_1 - data_target_class, axis=1))
    sample_2 = data_target_class[dists[1]].reshape(1, -1)
    two_samples = np.vstack((sample_1, sample_2))
    lmbda_vec = lmbda_vec_func().reshape(-1, 1)
    synthesis_data = np.dot(lmbda_vec.T, two_samples)
    return synthesis_data, target_class


def single_point_vicinal(data, labels, stddev, multiple, vicinal_type, augment=False):
    # Select distribution involving single data point #
    if vicinal_type == 'gaussian':
        vicinal = np.random.normal
    elif vicinal_type == 'laplace':
        vicinal = np.random.laplace

    # Single point vicinal perturbation with original_multiple #
    perturb_data = np.empty((0, data.shape[1]))
    perturb_labels = np.empty(0)
    for i in range(multiple):
        perturb_data = np.vstack((perturb_data, vicinal(data, stddev)))
        perturb_labels = np.hstack((perturb_labels, labels.copy()))

    # Augmentation vs Perturbation, and shuffle #
    if augment:
        aug_data = np.vstack((data, perturb_data))
        aug_labels = np.hstack((labels, perturb_labels))
        rnd_idx = np.random.permutation(aug_data.shape[0])
        aug_data = aug_data[rnd_idx]
        aug_labels = aug_labels[rnd_idx]
        return aug_data, aug_labels
    else:
        rnd_idx = np.random.permutation(perturb_data.shape[0])
        perturb_data = perturb_data[rnd_idx]
        perturb_labels = perturb_labels[rnd_idx]
        return perturb_data, perturb_labels


def convex_combination(data, labels, lmbda_vec_func, multiple, combine_type_func, augment=False):
    # convex combination perturbation with original_multiple #
    perturb_data = np.empty((0, data.shape[1]))
    perturb_labels = np.empty(0)
    for i in range(multiple):
        for j in range(data.shape[0]):
            synthesis_data, synthesis_label = combine_type_func(data, labels, lmbda_vec_func)
            perturb_data = np.vstack((perturb_data, synthesis_data))
            perturb_labels = np.hstack((perturb_labels, synthesis_label))

    # Augmentation vs Perturbation, and shuffle #
    if augment:
        aug_data = np.vstack((data, perturb_data))
        aug_labels = np.hstack((labels, perturb_labels))
        rnd_idx = np.random.permutation(aug_data.shape[0])
        aug_data = aug_data[rnd_idx]
        aug_labels = aug_labels[rnd_idx]
        return aug_data, aug_labels
    else:
        rnd_idx = np.random.permutation(perturb_data.shape[0])
        perturb_data = perturb_data[rnd_idx]
        perturb_labels = perturb_labels[rnd_idx]
        return perturb_data, perturb_labels
