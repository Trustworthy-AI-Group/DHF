import os
import numpy as np
import scipy.stats as st
from imageio import imread, imsave
import tensorflow as tf


def get_labels(names, f2l):
    labels = []
    for name in names:
        labels.append(f2l[name])
    return np.array(labels, dtype=np.int64)


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, "*")):
        with tf.gfile.Open(filepath, "rb") as f:
            image = imread(f, pilmode="RGB").astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), "w") as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format="png")


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_all_filenames(input_dir):
    filenames = []
    for filepath in tf.gfile.Glob(os.path.join(input_dir, "*")):
        filenames.append(os.path.basename(filepath))
    return filenames


def load_labels(file_name):
    import pandas as pd

    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]["filename"]: dev.iloc[i]["label"] for i in range(len(dev))}
    return f2l
