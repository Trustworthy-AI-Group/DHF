# coding=utf-8
"""Implementation of NI-FGSM attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np

import tensorflow as tf

from nets import (
    inception_resnet_v2_dhf,
    inception_v3,
    inception_v3_dhf,
    inception_v4,
    inception_resnet_v2,
    resnet_101_dhf,
    resnet_v2,
    resnet_152_dhf,
)

from utils import (
    check_or_create_dir,
    get_all_filenames,
    get_labels,
    load_images,
    save_images,
    load_labels,
)

slim = tf.contrib.slim

tf.flags.DEFINE_string("arch", "", "arch: inc_v3, inc_res_v2, res_101, res_152.")

tf.flags.DEFINE_string("method", "dhf", "method: original, dhf")

tf.flags.DEFINE_integer("batch_size", 10, "How many images process at one time.")

tf.flags.DEFINE_float("max_epsilon", 16.0, "max epsilon.")

tf.flags.DEFINE_integer("num_iter", 10, "max iteration.")

tf.flags.DEFINE_float(
    "mixup_weight_max",
    0.2,
    "the portion of the mixup logits is in the mixed feature map, i.e., /eta_max in the paper.",
)

tf.flags.DEFINE_float(
    "random_keep_prob",
    0.9,
    "the portion of the features that will NOT be randomly adjusted, i.e., 1-/rho.",
)

tf.flags.DEFINE_float("momentum", 1.0, "momentum about the model.")

tf.flags.DEFINE_integer("image_width", 299, "Width of each input images.")

tf.flags.DEFINE_integer("image_height", 299, "Height of each input images.")

tf.flags.DEFINE_float("prob", 0.5, "probability of using diverse inputs.")

tf.flags.DEFINE_string(
    "checkpoint_path", "./models", "Path to checkpoint for pretained models."
)

tf.flags.DEFINE_string("input_dir", "./dev_data/", "Input directory with images.")

tf.flags.DEFINE_string("output_dir", "./results", "Output directory with images.")

tf.flags.DEFINE_string("dhf_layers", "-1", "the layers will be applied to DHF")

FLAGS = tf.flags.FLAGS

np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)


arch_to_model_map = {
    "inc_v3": {
        "full_name": "inception_v3",
        "title": "InceptionV3",
        "model": inception_v3_dhf,
        "model_kernel": inception_v3_dhf.inception_v3,
        "arg_scope": inception_v3_dhf.inception_v3_arg_scope,
        "dhf_layers": inception_v3_dhf.dhf_layers,
    },
    "inc_res_v2": {
        "full_name": "inception_resnet_v2",
        "title": "InceptionResnetV2",
        "model": inception_resnet_v2_dhf,
        "model_kernel": inception_resnet_v2_dhf.inception_resnet_v2,
        "arg_scope": inception_resnet_v2_dhf.inception_resnet_v2_arg_scope,
        "dhf_layers": inception_resnet_v2_dhf.dhf_layers,
    },
    "res_101": {
        "full_name": "resnet_v2_101",
        "title": "resnet_v2",
        "model": resnet_101_dhf,
        "model_kernel": resnet_101_dhf.resnet_v2_101,
        "arg_scope": resnet_101_dhf.resnet_arg_scope,
        "dhf_layers": resnet_101_dhf.dhf_layers,
    },
    "res_152": {
        "full_name": "resnet_v2_152",
        "title": "resnet_v2",
        "model": resnet_152_dhf,
        "model_kernel": resnet_152_dhf.resnet_v2_152,
        "arg_scope": resnet_152_dhf.resnet_arg_scope,
        "dhf_layers": resnet_152_dhf.dhf_layers,
    },
}

arch = FLAGS.arch
model_name = arch_to_model_map[arch]["full_name"]
model_title = arch_to_model_map[arch]["title"]
model = arch_to_model_map[arch]["model"]
model_kernel = arch_to_model_map[arch]["model_kernel"]
model_arg_scope = arch_to_model_map[arch]["arg_scope"]


layer_idx_to_name_map = model.layer_idx_to_name_map


model_checkpoint_map = {
    "inception_v3": os.path.join(FLAGS.checkpoint_path, "inception_v3.ckpt"),
    "adv_inception_v3": os.path.join(
        FLAGS.checkpoint_path, "adv_inception_v3_rename.ckpt"
    ),
    "ens3_adv_inception_v3": os.path.join(
        FLAGS.checkpoint_path, "ens3_adv_inception_v3_rename.ckpt"
    ),
    "ens4_adv_inception_v3": os.path.join(
        FLAGS.checkpoint_path, "ens4_adv_inception_v3_rename.ckpt"
    ),
    "inception_v4": os.path.join(FLAGS.checkpoint_path, "inception_v4.ckpt"),
    "inception_resnet_v2": os.path.join(
        FLAGS.checkpoint_path, "inception_resnet_v2_2016_08_30.ckpt"
    ),
    "ens_adv_inception_resnet_v2": os.path.join(
        FLAGS.checkpoint_path, "ens_adv_inception_resnet_v2_rename.ckpt"
    ),
    "resnet_v2_101": os.path.join(FLAGS.checkpoint_path, "resnet_v2_101.ckpt"),
    "resnet_v2_152": os.path.join(FLAGS.checkpoint_path, "resnet_v2_152.ckpt"),
}


def get_mixup_features(mixup_end_points, if_mixup):
    """
    Function used to get the mixup_features, mixup_weight map.

    Args:
        mixup_end_points (_type_): the features of benign images.
        if_mixup (_type_): indicators indicating whether to do DHF for each image.

    Returns:
        _type_: a map: layer_idx -> (mixup_features, mixup_weight)
    """
    mixup_features = dict()
    dhf_layers = FLAGS.dhf_layers
    if dhf_layers == "-1":
        dhf_layers = arch_to_model_map[FLAGS.arch]["dhf_layers"]
    if isinstance(dhf_layers, str):
        # transfer the dhf_layers from string, like "1_2_3", to list, [1, 2, 3]
        dhf_layers = [int(idx) for idx in dhf_layers.split("_")]

    mixup_weight_max = FLAGS.mixup_weight_max

    if_mixup = tf.cast(if_mixup, tf.float32)

    for layer_idx in dhf_layers:
        if layer_idx > 0:
            weight = tf.random.uniform(
                shape=mixup_end_points[layer_idx_to_name_map[layer_idx]].shape,
                minval=0,
                maxval=mixup_weight_max,
            )
            mixup_features.update(
                {
                    layer_idx: {
                        "weight": weight * tf.reshape(if_mixup, [-1, 1, 1, 1]),
                        "feature_map": mixup_end_points[
                            layer_idx_to_name_map[layer_idx]
                        ],
                    }
                }
            )
    return mixup_features


def graph(x, x_mixup, y, i, x_max, x_min, grad, if_mixup):
    """
    Graph function.
    """
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    batch_size = FLAGS.batch_size

    # keep probabilities for DHF
    keep_prob_lower_bd = 1 - tf.cast(if_mixup, tf.float32)

    mixup_keep_prob = keep_prob_lower_bd

    random_keep_prob = tf.fill([batch_size], value=FLAGS.random_keep_prob)
    random_keep_prob = tf.maximum(random_keep_prob, keep_prob_lower_bd)
    x_nes = x + momentum * alpha * grad

    with slim.arg_scope(model_arg_scope()):
        logits_list = []
        if FLAGS.method == "dhf":
            # 1. get the features of benign images (mixup_features)
            mixup_features = dict()
            mixup_logits, mixup_end_points = model_kernel(
                x_mixup,
                method="original",
                num_classes=num_classes,
                is_training=False,
                reuse=tf.AUTO_REUSE,
            )
            mixup_features = get_mixup_features(mixup_end_points, if_mixup)

            # 2. attack with DHF using mixup_features
            logits_, end_points_ = model_kernel(
                x_nes,
                method=FLAGS.method,
                num_classes=num_classes,
                is_training=False,
                reuse=tf.AUTO_REUSE,
                mixup_features=mixup_features,
                mixup_keep_prob=mixup_keep_prob,
                random_keep_prob=random_keep_prob,
            )
        elif FLAGS.method == "original":
            logits_, end_points_ = model_kernel(
                x_nes,
                method=FLAGS.method,
                num_classes=num_classes,
                is_training=False,
                reuse=tf.AUTO_REUSE,
            )
        logits_list.append(logits_)

    logits_mean = tf.reduce_mean(logits_list, axis=0)
    pred = tf.argmax(logits_mean, axis=1)

    if_mixup |= tf.not_equal(pred, y)

    one_hot = tf.one_hot(y, num_classes)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits_mean)

    noise = tf.gradients(cross_entropy, x)[0]
    noise = noise / (tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True) + 1e-12)
    noise = momentum * grad + noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, x_mixup, y, i, x_max, x_min, noise, if_mixup


def stop(x, x_mixup, y, i, x_max, x_min, grad, if_mixup):
    """
    stop function.
    """
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def simple_eval():
    """
    Evaluation function.
    """
    f2l = load_labels(os.path.join(FLAGS.input_dir, "./val_rs.csv"))
    input_dir = FLAGS.output_dir
    print(input_dir)

    batch_shape = [50, 299, 299, 3]
    num_classes = 1001
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False
            )

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
                x_input,
                num_classes=num_classes,
                is_training=False,
                scope="AdvInceptionV3",
            )

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                x_input,
                num_classes=num_classes,
                is_training=False,
                scope="Ens3AdvInceptionV3",
            )

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                x_input,
                num_classes=num_classes,
                is_training=False,
                scope="Ens4AdvInceptionV3",
            )

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_v4, end_points_v4 = inception_v4.inception_v4(
                x_input, num_classes=num_classes, is_training=False
            )

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False
            )

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            (
                logits_ens_adv_res_v2,
                end_points_ens_adv_res_v2,
            ) = inception_resnet_v2.inception_resnet_v2(
                x_input,
                num_classes=num_classes,
                is_training=False,
                scope="EnsAdvInceptionResnetV2",
            )

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet_101, end_points_resnet_101 = resnet_v2.resnet_v2_101(
                x_input,
                num_classes=num_classes,
                is_training=False,
                scope="resnet_v2_101",
            )

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet_152, end_points_resnet_152 = resnet_v2.resnet_v2_152(
                x_input,
                num_classes=num_classes,
                is_training=False,
                scope="resnet_v2_152",
            )

        pred_v3 = tf.argmax(end_points_v3["Predictions"], 1)
        pred_adv_v3 = tf.argmax(end_points_adv_v3["Predictions"], 1)
        pred_ens3_adv_v3 = tf.argmax(end_points_ens3_adv_v3["Predictions"], 1)
        pred_ens4_adv_v3 = tf.argmax(end_points_ens4_adv_v3["Predictions"], 1)
        pred_v4 = tf.argmax(end_points_v4["Predictions"], 1)
        pred_res_v2 = tf.argmax(end_points_res_v2["Predictions"], 1)
        pred_ens_adv_res_v2 = tf.argmax(end_points_ens_adv_res_v2["Predictions"], 1)
        pred_resnet_101 = tf.argmax(
            end_points_resnet_101["predictions"], 1
        )  # resnet 101
        pred_resnet_152 = tf.argmax(
            end_points_resnet_152["predictions"], 1
        )  # resnet 152

        s1 = tf.train.Saver(slim.get_model_variables(scope="InceptionV3"))
        s2 = tf.train.Saver(slim.get_model_variables(scope="AdvInceptionV3"))
        s3 = tf.train.Saver(slim.get_model_variables(scope="Ens3AdvInceptionV3"))
        s4 = tf.train.Saver(slim.get_model_variables(scope="Ens4AdvInceptionV3"))
        s5 = tf.train.Saver(slim.get_model_variables(scope="InceptionV4"))
        s6 = tf.train.Saver(slim.get_model_variables(scope="InceptionResnetV2"))
        s7 = tf.train.Saver(slim.get_model_variables(scope="EnsAdvInceptionResnetV2"))
        s8 = tf.train.Saver(
            slim.get_model_variables(scope="resnet_v2_101")
        )  # resnet 101
        s9 = tf.train.Saver(
            slim.get_model_variables(scope="resnet_v2_152")
        )  # resnet 152

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map["inception_v3"])
            s2.restore(sess, model_checkpoint_map["adv_inception_v3"])
            s3.restore(sess, model_checkpoint_map["ens3_adv_inception_v3"])
            s4.restore(sess, model_checkpoint_map["ens4_adv_inception_v3"])
            s5.restore(sess, model_checkpoint_map["inception_v4"])
            s6.restore(sess, model_checkpoint_map["inception_resnet_v2"])
            s7.restore(sess, model_checkpoint_map["ens_adv_inception_resnet_v2"])
            s8.restore(sess, model_checkpoint_map["resnet_v2_101"])  # resnet 101
            s9.restore(sess, model_checkpoint_map["resnet_v2_152"])  # resnet 152

            model_names = [
                "inception_v3",
                "inception_v4",
                "inception_resnet_v2",
                "resnet_101",
                "resnet_152",
                "ens3_adv_inception_v3",
                "ens4_adv_inception_v3",
                "ens_adv_inception_resnet_v2",
                "adv_inception_v3",
            ]
            success_count = np.zeros(len(model_names))

            idx = 0
            for filenames, images in load_images(input_dir, batch_shape):
                idx += 1
                print(f"start the i={idx} eval")
                (
                    v3,
                    adv_v3,
                    ens3_adv_v3,
                    ens4_adv_v3,
                    v4,
                    res_v2,
                    ens_adv_res_v2,
                    resnet_101,
                    resnet_152,
                ) = sess.run(
                    (
                        pred_v3,
                        pred_adv_v3,
                        pred_ens3_adv_v3,
                        pred_ens4_adv_v3,
                        pred_v4,
                        pred_res_v2,
                        pred_ens_adv_res_v2,
                        pred_resnet_101,
                        pred_resnet_152,
                    ),
                    feed_dict={x_input: images},
                )

                for filename, l1, l2, l3, l4, l5, l6, l7, l8, l9 in zip(
                    filenames,
                    v3,
                    adv_v3,
                    ens3_adv_v3,
                    ens4_adv_v3,
                    v4,
                    res_v2,
                    ens_adv_res_v2,
                    resnet_101,
                    resnet_152,
                ):
                    label = f2l[filename]
                    l = [l1, l5, l6, l8, l9, l3, l4, l7, l2]
                    for i in range(len(model_names)):
                        if l[i] != label:
                            success_count[i] += 1
    for i, name in enumerate(model_names):
        print(
            f"Attack Success Rate for {name} : {success_count[i] / 1000.0 * 100:.1f}%"
        )
    print(np.array(success_count, dtype=np.float32) / 1000.0 * 100)


def main(_):
    """
    main function
    """
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    f2l = load_labels(os.path.join(FLAGS.input_dir, "./val_rs.csv"))
    eps = 2 * FLAGS.max_epsilon / 255.0

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_mixup = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.placeholder(tf.int64, shape=[FLAGS.batch_size])
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        if_mixup = tf.cast(tf.zeros(shape=[FLAGS.batch_size]), tf.bool)

        x_adv, _, _, _, _, _, _, _ = tf.while_loop(
            stop, graph, [x_input, x_mixup, y, i, x_max, x_min, grad, if_mixup]
        )

        # Run computation
        s0 = tf.train.Saver(slim.get_model_variables(scope=model_title))
        # s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        # s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        # s6 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        # s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        # s8 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            s0.restore(sess, model_checkpoint_map[model_name])
            # s1.restore(sess, model_checkpoint_map['inception_v3'])
            # s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2'])
            # s5.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            # s6.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            # s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            # s8.restore(sess, model_checkpoint_map['adv_inception_v3'])

            idx = 0
            l2_diff = 0
            images_path = os.path.join(FLAGS.input_dir, "./val_rs")
            all_filenames = get_all_filenames(images_path)
            np.random.shuffle(all_filenames)
            for filenames, images in load_images(images_path, batch_shape):
                idx = idx + 1
                print(f"start the i={idx} attack")

                x_mixup_input = images

                labels = get_labels(filenames, f2l)
                adv_images = sess.run(
                    x_adv,
                    feed_dict={x_input: images, x_mixup: x_mixup_input, y: labels},
                )
                save_images(adv_images, filenames, FLAGS.output_dir)
                diff = (adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255
                l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))

            print(f"mean l2 diff: {l2_diff * FLAGS.batch_size / 1000:.2f}")
    print(f"Save adv images to {FLAGS.output_dir}")
    simple_eval()


if __name__ == "__main__":
    tf.app.run()
