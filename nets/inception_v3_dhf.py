# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import inception_utils

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def DHF(net, 
        mixup_feature, mixup_weight, mixup_keep_prob, 
        random_keep_prob=0.994
      ):
  with tf.name_scope('dhf') as scope:
    # 1. mixup
    mixup_mask = tf.cast((tf.random_uniform(net.shape, minval=0, maxval=1.0) > tf.reshape(mixup_keep_prob, [-1, 1, 1, 1])), dtype=tf.float32)
    # net = mixup_mask * (tf.reshape(mixup_weight, [-1, 1, 1, 1]) * mixup_feature + (1 - tf.reshape(mixup_weight, [-1, 1, 1, 1])) * net) + (1 - mixup_mask) * net
    net = mixup_mask * (mixup_weight * mixup_feature + (1 - mixup_weight) * net) + (1 - mixup_mask) * net

    # 2. random
    random_mask = tf.cast(tf.random_uniform(net.shape, minval=0, maxval=1.0)>tf.reshape(random_keep_prob, [-1, 1, 1, 1]), dtype=tf.float32)
    random_val = tf.math.reduce_mean(net, axis=[1, 2, 3], keep_dims=True)
    net = (1 - random_mask) * net + random_mask * random_val
    return net


dhf_layers = "12_13_15_16_17_19_20_21_23_24_25_26_27_29_30_31_32_33_35_36_37_38_39_41_42_43_44_45_47_48_49_50_52_53_54_56_57_58_60"


layer_idx_to_name_map = {
  1: "Conv2d_1a_3x3", # 1
  2: "Conv2d_2a_3x3", # 2
  3: "Conv2d_2b_3x3", # 3
  4: "MaxPool_3a_3x3", # 4
  5: "Conv2d_3b_1x1", # 5
  6: "Conv2d_4a_3x3", # 6
  7: "MaxPool_5a_3x3",   # 7
  8: "Mixed_5b_B2_conv1_8", # 8 
  9: "Mixed_5b_B2_conv3_9", # 9
  10: "Mixed_5b_B2_conv3_10", # 
  11: "Mixed_5b",   # 10
  12: "Mixed_5c_B2_conv1_12", # 11
  13: "Mixed_5c_B2_conv3_13", # 12
  14: "Mixed_5c_B2_conv3_14", # 
  15: "Mixed_5c",   # 13
  16: "Mixed_5d_B2_conv1_16", # 14
  17: "Mixed_5d_B2_conv3_17", # 15
  18: "Mixed_5d_B2_conv3_18", # 
  19: "Mixed_5d",   # 16
  20: "Mixed_6a_B1_conv1_20", # 17
  21: "Mixed_6a_B1_conv3_21", # 18
  22: "Mixed_6a_B1_conv1_22", # 
  23: "Mixed_6a",   # 19
  24: "Mixed_6b_B2_conv1_24",    # 20
  25: "Mixed_6b_B2_conv7x1_25",  # 21
  26: "Mixed_6b_B2_conv1x7_26",  # 22
  27: "Mixed_6b_B2_conv7x1_27",  # 23
  28: "Mixed_6b_B2_conv1x7_28",  # 
  29: "Mixed_6b",   # 24
  30: "Mixed_6c_B2_conv1_30",    # 25
  31: "Mixed_6c_B2_conv7x1_31",  # 26
  32: "Mixed_6c_B2_conv1x7_32",  # 27
  33: "Mixed_6c_B2_conv7x1_33",  # 28
  34: "Mixed_6c_B2_conv1x7_34",  # 
  35: "Mixed_6c",   # 29
  36: "Mixed_6d_B2_conv1_36",    # 30
  37: "Mixed_6d_B2_conv7x1_37",  # 31
  38: "Mixed_6d_B2_conv1x7_38",  # 32
  39: "Mixed_6d_B2_conv7x1_39",  # 33
  40: "Mixed_6d_B2_conv1x7_40",  # 
  41: "Mixed_6d",   # 34
  42: "Mixed_6e_B2_conv1_42",    # 35
  43: "Mixed_6e_B2_conv7x1_43",  # 36
  44: "Mixed_6e_B2_conv1x7_44",  # 37
  45: "Mixed_6e_B2_conv7x1_45",  # 38
  46: "Mixed_6e_B2_conv1x7_46",  # 
  47: "Mixed_6e",   # 39
  48: "Mixed_7a_B1_conv1_48",    # 40
  49: "Mixed_7a_B1_conv1x7_49",  # 41
  50: "Mixed_7a_B1_conv7x1_50",  # 42
  51: "Mixed_7a_B1_conv3_51",    # 
  52: "Mixed_7a",   # 43
  53: "Mixed_7b_B2_conv1_53",    # 44
  54: "Mixed_7b_B2_conv3_54",    # 45
  55: "Mixed_7b_B2_conv_cat_55", # 
  56: "Mixed_7b",   # 46
  57: "Mixed_7c_B2_conv1_57",    # 48
  58: "Mixed_7c_B2_conv3_58",    # 49
  59: "Mixed_7c_B2_conv_cat_59", # 
  60: "Mixed_7c"    # 50
}


layer_idx_to_name_map2 = {
  1: "Conv2d_1a_3x3", # 1
  2: "Conv2d_2a_3x3", # 2
  3: "Conv2d_2b_3x3", # 3
  4: "MaxPool_3a_3x3", # 4
  5: "Conv2d_3b_1x1", # 5
  6: "Conv2d_4a_3x3", # 6
  7: "MaxPool_5a_3x3",   # 7
  8: "Mixed_5b_B2_conv1_8", # 8 
  9: "Mixed_5b_B2_conv3_9", # 9
  10: "Mixed_5b_B2_conv3_10", # 10
  11: "Mixed_5b",   # 
  12: "Mixed_5c_B2_conv1_12", # 11
  13: "Mixed_5c_B2_conv3_13", # 12
  14: "Mixed_5c_B2_conv3_14", # 13
  15: "Mixed_5c",   # 
  16: "Mixed_5d_B2_conv1_16", # 14
  17: "Mixed_5d_B2_conv3_17", # 15
  18: "Mixed_5d_B2_conv3_18", # 16
  19: "Mixed_5d",   # 
  20: "Mixed_6a_B1_conv1_20", # 17
  21: "Mixed_6a_B1_conv3_21", # 18
  22: "Mixed_6a_B1_conv1_22", # 19
  23: "Mixed_6a",   # 
  24: "Mixed_6b_B2_conv1_24",    # 20
  25: "Mixed_6b_B2_conv7x1_25",  # 21
  26: "Mixed_6b_B2_conv1x7_26",  # 22
  27: "Mixed_6b_B2_conv7x1_27",  # 23
  28: "Mixed_6b_B2_conv1x7_28",  # 24
  29: "Mixed_6b",   # 
  30: "Mixed_6c_B2_conv1_30",    # 25
  31: "Mixed_6c_B2_conv7x1_31",  # 26
  32: "Mixed_6c_B2_conv1x7_32",  # 27
  33: "Mixed_6c_B2_conv7x1_33",  # 28
  34: "Mixed_6c_B2_conv1x7_34",  # 29
  35: "Mixed_6c",   # 
  36: "Mixed_6d_B2_conv1_36",    # 30
  37: "Mixed_6d_B2_conv7x1_37",  # 31
  38: "Mixed_6d_B2_conv1x7_38",  # 32
  39: "Mixed_6d_B2_conv7x1_39",  # 33
  40: "Mixed_6d_B2_conv1x7_40",  # 34
  41: "Mixed_6d",   # 
  42: "Mixed_6e_B2_conv1_42",    # 35
  43: "Mixed_6e_B2_conv7x1_43",  # 36
  44: "Mixed_6e_B2_conv1x7_44",  # 37
  45: "Mixed_6e_B2_conv7x1_45",  # 38
  46: "Mixed_6e_B2_conv1x7_46",  # 39
  47: "Mixed_6e",   # 
  48: "Mixed_7a_B1_conv1_48",    # 40
  49: "Mixed_7a_B1_conv1x7_49",  # 41
  50: "Mixed_7a_B1_conv7x1_50",  # 42
  51: "Mixed_7a_B1_conv3_51",    # 43
  52: "Mixed_7a",   # 
  53: "Mixed_7b_B2_conv1_53",    # 44
  54: "Mixed_7b_B2_conv3_54",    # 45
  55: "Mixed_7b_B2_conv_cat_55", # 46
  56: "Mixed_7b",   # 
  57: "Mixed_7c_B2_conv1_57",    # 48
  58: "Mixed_7c_B2_conv3_58",    # 49
  59: "Mixed_7c_B2_conv_cat_59", # 50
  60: "Mixed_7c"    # 
}


def inception_v3_base(inputs,
                      mixup_features,
                      method=None,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None,
                      mixup_keep_prob=tf.fill([1, 1], value=0.8),
                      random_keep_prob=tf.fill([1, 1], value=0.8)):
  """Inception model from http://arxiv.org/abs/1512.00567.

  Constructs an Inception v3 network from inputs to the given final endpoint.
  This method can construct the network up to the final inception block
  Mixed_7c.

  Note that the names of the layers in the paper do not correspond to the names
  of the endpoints registered by this function although they build the same
  network.

  Here is a mapping from the old_names to the new names:
  No.       | Old name          | New name
  =======================================
  1         | conv0             | Conv2d_1a_3x3
  2         | conv1             | Conv2d_2a_3x3
  3         | conv2             | Conv2d_2b_3x3
  4         | pool1             | MaxPool_3a_3x3
  5         | conv3             | Conv2d_3b_1x1
  6         | conv4             | Conv2d_4a_3x3
  7         | pool2             | MaxPool_5a_3x3
  8         | mixed_35x35x256a  | Mixed_5b
  9         | mixed_35x35x288a  | Mixed_5c
  10        | mixed_35x35x288b  | Mixed_5d
  11        | mixed_17x17x768a  | Mixed_6a
  12        | mixed_17x17x768b  | Mixed_6b
  13        | mixed_17x17x768c  | Mixed_6c
  14        | mixed_17x17x768d  | Mixed_6d
  15        | mixed_17x17x768e  | Mixed_6e
  16        | mixed_8x8x1280a   | Mixed_7a
  17        | mixed_8x8x2048a   | Mixed_7b
  18        | mixed_8x8x2048b   | Mixed_7c

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

  def our_method(net):
    net = DHF(net, 
              mixup_feature=mixup_features[layer_idx]["feature_map"], 
              mixup_weight=mixup_features[layer_idx]["weight"], 
              mixup_keep_prob=mixup_keep_prob,
              random_keep_prob=random_keep_prob
            )
    return net

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  layer_idx = 1

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID'):
      # 299 x 299 x 3
      end_point = 'Conv2d_1a_3x3'  # 1
      net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # 149 x 149 x 32
      end_point = 'Conv2d_2a_3x3'  # 2
      net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # 147 x 147 x 32
      end_point = 'Conv2d_2b_3x3'  # 3
      net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # 147 x 147 x 64
      end_point = 'MaxPool_3a_3x3'  # 4
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # 73 x 73 x 64
      end_point = 'Conv2d_3b_1x1'  # 5
      net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # 73 x 73 x 80.
      end_point = 'Conv2d_4a_3x3'  # 6
      net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # 71 x 71 x 192.
      end_point = 'MaxPool_5a_3x3'  # 7
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1
      # 35 x 35 x 192.

    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # mixed: 35 x 35 x 256.
      end_point = 'Mixed_5b'  # 8
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_5b_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          end_points["Mixed_5b_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
          end_points["Mixed_5b_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1
  
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_5b'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed_1: 35 x 35 x 288.
      end_point = 'Mixed_5c'  # 9
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                 scope='Conv2d_0a_1x1')
          end_points["Mixed_5c_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          end_points["Mixed_5c_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
          end_points["Mixed_5c_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_5c'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed_2: 35 x 35 x 288.
      end_point = 'Mixed_5d'  # 10
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_5d_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          end_points["Mixed_5d_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
          end_points["Mixed_5d_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed_3: 17 x 17 x 768.
      end_point = 'Mixed_6a'  # 11
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_6a_B1_conv1_"+str(layer_idx)] = branch_1
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_1 = our_method(branch_1)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          end_points["Mixed_6a_B1_conv3_"+str(layer_idx)] = branch_1
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_1 = our_method(branch_1)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
          end_points["Mixed_6a_B1_conv1_"+str(layer_idx)] = branch_1
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_1 = our_method(branch_1)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points['Mixed_6a'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed4: 17 x 17 x 768.
      end_point = 'Mixed_6b'  # 12
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_6b_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0b_7x1')
          end_points["Mixed_6b_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                 scope='Conv2d_0c_1x7')
          end_points["Mixed_6b_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0d_7x1')
          end_points["Mixed_6b_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
          end_points["Mixed_6b_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_6b'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1


      # mixed_5: 17 x 17 x 768.
      end_point = 'Mixed_6c'  # 13
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_6c_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          end_points["Mixed_6c_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1
                      
          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          end_points["Mixed_6c_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          end_points["Mixed_6c_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
          end_points["Mixed_6c_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_6c'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed_6: 17 x 17 x 768.
      end_point = 'Mixed_6d'  # 14
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_6d_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          end_points["Mixed_6d_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          end_points["Mixed_6d_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          end_points["Mixed_6d_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
          end_points["Mixed_6d_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_6d'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1


      # mixed_7: 17 x 17 x 768.
      end_point = 'Mixed_6e'  # 15
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_6e_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0b_7x1')
          end_points["Mixed_6e_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0c_1x7')
          end_points["Mixed_6e_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0d_7x1')
          end_points["Mixed_6e_B2_conv7x1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
          end_points["Mixed_6e_B2_conv1x7_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_6e'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed_8: 8 x 8 x 1280.
      end_point = 'Mixed_7a'  # 16
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_7a_B1_conv1_"+str(layer_idx)] = branch_1
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_1 = our_method(branch_1)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          end_points["Mixed_7a_B1_conv1x7_"+str(layer_idx)] = branch_1
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_1 = our_method(branch_1)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
          end_points["Mixed_7a_B1_conv7x1_"+str(layer_idx)] = branch_1
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_1 = our_method(branch_1)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
          end_points["Mixed_7a_B1_conv3_"+str(layer_idx)] = branch_1
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_1 = our_method(branch_1)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points['Mixed_7a'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed_9: 8 x 8 x 2048.
      end_point = 'Mixed_7b'  # 17
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_7b_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          end_points["Mixed_7b_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
          end_points["Mixed_7b_B2_conv_cat_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_7b'] = net
      if end_point == final_endpoint: 
        return net, end_points
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1

      # mixed_10: 8 x 8 x 2048.
      end_point = 'Mixed_7c'  # 18
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          end_points["Mixed_7c_B2_conv1_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          end_points["Mixed_7c_B2_conv3_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1

          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
          end_points["Mixed_7c_B2_conv_cat_"+str(layer_idx)] = branch_2
          if method == "dhf" and layer_idx in mixup_features.keys():
            branch_2 = our_method(branch_2)
            print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
          layer_idx += 1
          
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points['Mixed_7c'] = net
      if method == "dhf" and layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(layer_idx, layer_idx_to_name_map[layer_idx]))
      layer_idx += 1
      if end_point == final_endpoint: 
        return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v3(inputs,
                 mixup_features=dict({}),
                 method=None,
                 num_classes=1000,
                 is_training=True,
                 final_endpoint='Mixed_7c',
                 dropout_keep_prob=0.8,
                 mixup_keep_prob=tf.fill([1, 1], value=0.8),
                 random_keep_prob=tf.fill([1, 1], value=0.8),
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 create_aux_logits=True,
                 scope='InceptionV3'):
  """Inception model from http://arxiv.org/abs/1512.00567.

  "Rethinking the Inception Architecture for Computer Vision"

  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna.

  With the default arguments this method constructs the exact model defined in
  the paper. However, one can experiment with variations of the inception_v3
  network by changing arguments dropout_keep_prob, min_depth and
  depth_multiplier.

  The default image size used to train this network is 299x299.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
        shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    create_aux_logits: Whether to create the auxiliary logits.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v3_base(
          inputs, mixup_features, method=method, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier,
          mixup_keep_prob=mixup_keep_prob, 
          random_keep_prob=random_keep_prob)
      # Auxiliary Head logits
      if create_aux_logits:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
          aux_logits = end_points['Mixed_6e']
          with tf.variable_scope('AuxLogits'):
            aux_logits = slim.avg_pool2d(
                aux_logits, [5, 5], stride=3, padding='VALID',
                scope='AvgPool_1a_5x5')
            aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],
                                     scope='Conv2d_1b_1x1')

            # Shape of feature map before the final layer.
            kernel_size = _reduced_kernel_size_for_small_input(
                aux_logits, [5, 5])
            aux_logits = slim.conv2d(
                aux_logits, depth(768), kernel_size,
                weights_initializer=trunc_normal(0.01),
                padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
            aux_logits = slim.conv2d(
                aux_logits, num_classes, [1, 1], activation_fn=None,
                normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                scope='Conv2d_2b_1x1')
            if spatial_squeeze:
              aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
            end_points['AuxLogits'] = aux_logits

      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
        net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                              scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        # 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v3.default_image_size = 299


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.

  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])

  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


inception_v3_arg_scope = inception_utils.inception_arg_scope
