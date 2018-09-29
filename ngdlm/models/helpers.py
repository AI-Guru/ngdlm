from keras import backend as K
import os
import numpy as np


def euclidean_loss(left, right):
    """
    Computes the euclidean loss.
    """

    distance = K.sum(K.square(left - right), axis=1)
    return distance


def cosine_loss(left, right):
    """
    Computes the cosine loss.
    """

    left = K.l2_normalize(left, axis=-1)
    right = K.l2_normalize(right, axis=-1)

    distance = K.constant(1.0) - K.batch_dot(left, right, axes=1)
    distance = K.squeeze(distance, axis=-1)

    return distance


def compute_latent_extremum(latent_sample, latent_samples, extremum_type, norm):
    """
    Computes the latent extremum.
    Used for sampling in triplet-loss training.
    """
    distances = [compute_latent_distance(latent_sample, l, norm) for l in latent_samples]
    if extremum_type == "argmax":
        return np.argmax(distances)
    elif extremum_type == "argmin":
        return np.argmin(distances)
    else:
        raise Exception("Unexpected: " + str(extremum_type))


def compute_latent_distance(latent_sample1, latent_sample2, norm):
    """
    Computes distances in latent space.
    """
    if norm == "euclidean":
        distance = np.sum(np.square(latent_sample2 - latent_sample1))
        return distance
    #elif norm == "cosine":
    #    distance = np.sum(np.square(latent_sample2 - latent_sample1))
    #    return distance
    else:
        raise Exception("Unexpected norm: " + norm)


def append_to_filepath(filepath, string):
    """
    Adds a string to a file-path. Right before the extension.
    """

    filepath, extension = os.path.splitext(filepath)
    return filepath + string + extension
