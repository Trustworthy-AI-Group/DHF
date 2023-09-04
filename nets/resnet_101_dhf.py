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
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from nets import resnet_utils

slim = tf.contrib.slim
# resnet_arg_scope = resnet_utils.resnet_arg_scope

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
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import collections
# import tensorflow as tf

# slim = tf.contrib.slim

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


dhf_layers = "26_27_28_31_32_33_36_37_38_41_42_43_46_47_48_51_52_53_56_57_58_61_62_63_66_67_68_71_72_73_76_77_78_81_82_83_86_87_88_91_92_93_96_97_98_101_102_103_106_107_108_111_112_113_116_117_118_121_122_123_126_127_128_131_132_133_136_137_138_141_142_143_146_147_148_151_152_153_156_157_158_161_162_163"


layer_idx_to_name_map = {
    1: 'conv1_1',   # # 1
    2: 'conv3_2',   # # 2
    3: 'conv1_3',   # # 3
    4: 'shortcut_0',   # 
    5: 'out_5',   # 
    6: 'conv1_6',   # # 4 
    7: 'conv3_7',   # # 5
    8: 'conv1_8',   # # 6
    9: 'shortcut_1',   # 
    10: 'out_10',   # 
    11: 'conv1_11',   # # 7
    12: 'conv3_12',   # # 8
    13: 'conv1_13',   # # 9
    14: 'shortcut_2',   # 
    15: 'out_15',   # 
    16: 'conv1_16',   # # 10
    17: 'conv3_17',   # # 11
    18: 'conv1_18',   # # 12
    19: 'shortcut_3',   # 
    20: 'out_20',   # 
    21: 'conv1_21',   # # 13
    22: 'conv3_22',   # # 14
    23: 'conv1_23',   # # 15
    24: 'shortcut_4',   # 
    25: 'out_25',   # 
    26: 'conv1_26',   # # 16
    27: 'conv3_27',   # # 17
    28: 'conv1_28',   # # 18
    29: 'shortcut_5',   # 
    30: 'out_30',   # 
    31: 'conv1_31',   # # 19
    32: 'conv3_32',   # # 20
    33: 'conv1_33',   # # 21
    34: 'shortcut_6',   # 
    35: 'out_35',   # 
    36: 'conv1_36',   # # 22
    37: 'conv3_37',   # # 23
    38: 'conv1_38',   # # 24
    39: 'shortcut_7',   # 
    40: 'out_40',   # 
    41: 'conv1_41',   # # 25
    42: 'conv3_42',   # # 26
    43: 'conv1_43',   # # 27
    44: 'shortcut_8',   # 
    45: 'out_45',   # 
    46: 'conv1_46',   # # 28
    47: 'conv3_47',   # # 29
    48: 'conv1_48',   # # 30
    49: 'shortcut_9',   # 
    50: 'out_50',   # 
    51: 'conv1_51',   # # 31
    52: 'conv3_52',   # # 32
    53: 'conv1_53',   # # 33
    54: 'shortcut_10',   # 
    55: 'out_55',   # 
    56: 'conv1_56',   # # 34
    57: 'conv3_57',   # # 35
    58: 'conv1_58',   # # 36
    59: 'shortcut_11',   # 
    60: 'out_60',   # 
    61: 'conv1_61',   # # 37
    62: 'conv3_62',   # # 38
    63: 'conv1_63',   # # 39
    64: 'shortcut_12',   # 
    65: 'out_65',   # 
    66: 'conv1_66',   # # 40
    67: 'conv3_67',   # # 41
    68: 'conv1_68',   # # 42
    69: 'shortcut_13',   # 
    70: 'out_70',   # 
    71: 'conv1_71',   # # 43
    72: 'conv3_72',   # # 44
    73: 'conv1_73',   # # 45
    74: 'shortcut_14',   # 
    75: 'out_75',   # 
    76: 'conv1_76',   # # 46
    77: 'conv3_77',   # # 47
    78: 'conv1_78',   # # 48
    79: 'shortcut_15',   # 
    80: 'out_80',   # 
    81: 'conv1_81',   # # 49
    82: 'conv3_82',   # # 50
    83: 'conv1_83',   # # 51
    84: 'shortcut_16',   # 
    85: 'out_85',   # 
    86: 'conv1_86',   # # 52
    87: 'conv3_87',   # # 53
    88: 'conv1_88',   # # 54
    89: 'shortcut_17',   # 
    90: 'out_90',   # 
    91: 'conv1_91',   # # 55
    92: 'conv3_92',   # # 56
    93: 'conv1_93',   # # 57
    94: 'shortcut_18',   # 
    95: 'out_95',   # 
    96: 'conv1_96',   # # 58
    97: 'conv3_97',   # # 59
    98: 'conv1_98',   # # 60
    99: 'shortcut_19',   # 
    100: 'out_100',   # 
    101: 'conv1_101',   # # 61
    102: 'conv3_102',   # # 62
    103: 'conv1_103',   # # 63
    104: 'shortcut_20',   # 
    105: 'out_105',   # 
    106: 'conv1_106',   # # 64
    107: 'conv3_107',   # # 65
    108: 'conv1_108',   # # 66
    109: 'shortcut_21',   # 
    110: 'out_110',   # 
    111: 'conv1_111',   # # 67
    112: 'conv3_112',   # # 68
    113: 'conv1_113',   # # 69
    114: 'shortcut_22',   # 
    115: 'out_115',   # 
    116: 'conv1_116',   # # 70
    117: 'conv3_117',   # # 71
    118: 'conv1_118',   # # 72
    119: 'shortcut_23',   # 
    120: 'out_120',   # 
    121: 'conv1_121',   # # 73
    122: 'conv3_122',   # # 74
    123: 'conv1_123',   # # 75
    124: 'shortcut_24',   # 
    125: 'out_125',   # 
    126: 'conv1_126',   # # 76
    127: 'conv3_127',   # # 77
    128: 'conv1_128',   # # 78
    129: 'shortcut_25',   # 
    130: 'out_130',   # 
    131: 'conv1_131',   # # 79
    132: 'conv3_132',   # # 80
    133: 'conv1_133',   # # 81
    134: 'shortcut_26',   # 
    135: 'out_135',   # 
    136: 'conv1_136',   # # 82
    137: 'conv3_137',   # # 83
    138: 'conv1_138',   # # 84
    139: 'shortcut_27',   # 
    140: 'out_140',   # 
    141: 'conv1_141',   # # 85
    142: 'conv3_142',   # # 86
    143: 'conv1_143',   # # 87
    144: 'shortcut_28',   # 
    145: 'out_145',   # 
    146: 'conv1_146',   # # 88
    147: 'conv3_147',   # # 89
    148: 'conv1_148',   # # 90
    149: 'shortcut_29',   # 
    150: 'out_150',   # 
    151: 'conv1_151',   # # 91
    152: 'conv3_152',   # # 92
    153: 'conv1_153',   # # 93
    154: 'shortcut_30',   # 
    155: 'out_155',   # 
    156: 'conv1_156',   # # 94
    157: 'conv3_157',   # # 95
    158: 'conv1_158',   # # 96
    159: 'shortcut_31',   # 
    160: 'out_160',   # 
    161: 'conv1_161',   # # 97
    162: 'conv3_162',   # # 98
    163: 'conv1_163',   # # 99
    164: 'shortcut_32',   # 
    165: 'out_165'  # 
}


shortcut_cnt = 0
g_layer_idx = 0
necessary_endpoints = dict({})
g_method = ""
g_mixup_features = dict({})
g_mixup_keep_prob = []
g_random_keep_prob = []

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
      network stride. If output_stride is not `None`, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of the ResNet.
      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1


  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
            rate *= unit.get('stride', 1)

          else:
            net = block.unit_fn(net, rate=1, **unit)
            current_stride *= unit.get('stride', 1)
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  global g_layer_idx
  global shortcut_cnt
  global g_method
  global g_mixup_features
  global g_mixup_keep_prob
  global g_random_keep_prob
  method = g_method
  mixup_features = g_mixup_features
  mixup_keep_prob = g_mixup_keep_prob
  random_keep_prob = g_random_keep_prob


  def our_method(net):
    global g_layer_idx
    layer_idx = g_layer_idx
    net = DHF(net, 
              mixup_feature=mixup_features[layer_idx]["feature_map"],
              mixup_weight=mixup_features[layer_idx]["weight"], 
              mixup_keep_prob=mixup_keep_prob,
              random_keep_prob=random_keep_prob
            )
    return net


  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')


    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    necessary_endpoints["conv1_"+str(g_layer_idx)] = residual
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      residual = our_method(residual)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    necessary_endpoints["conv3_"+str(g_layer_idx)] = residual
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      residual = our_method(residual)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')
    necessary_endpoints["conv1_"+str(g_layer_idx)] = residual
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      residual = our_method(residual)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    necessary_endpoints["shortcut_"+str(shortcut_cnt)] = shortcut
    if method == "original":
      output = shortcut + residual
    elif method == "dhf":
      if g_layer_idx in mixup_features.keys():
        shortcut = our_method(shortcut)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
        output = shortcut + residual
      else:
        output = shortcut + residual
    else:
      raise("raise no method '{}'".format(method))
    g_layer_idx += 1
    shortcut_cnt += 1
  
    necessary_endpoints["out_"+str(g_layer_idx)] = output
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      output = our_method(output)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1  

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


def resnet_v2(inputs,
              blocks,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              mixup_features=dict({}),
              method=None,
              num_classes=1001,
              is_training=True,
              mixup_keep_prob=[0.8]*55,
              random_keep_prob=[0.8]*55,
              spatial_squeeze=True,
              reuse=None,
              scope=None,
              ):
  """Generator for v2 (preactivation) ResNet models.

  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
        To use this parameter, the input images must be smaller than 300x300
        pixels, in which case the output logit layer does not contain spatial
        information and can be removed.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.


  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  end_points = {}
  global g_layer_idx
  g_layer_idx = 1
  global shortcut_cnt
  shortcut_cnt = 0
  global necessary_endpoints
  necessary_endpoints = dict({})
  global g_method
  global g_mixup_features
  global g_mixup_keep_prob
  global g_random_keep_prob
  g_method = method
  g_mixup_features = mixup_features
  g_mixup_keep_prob = mixup_keep_prob
  g_random_keep_prob = random_keep_prob

  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')


        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        # print(end_points.keys())
        # print(necessary_endpoints.keys())
        if num_classes is not None:
          end_points['predictions'] = slim.softmax(net, scope='predictions')
          necessary_endpoints["predictions"] = end_points["predictions"]
        # return net, end_points
        return net, necessary_endpoints


resnet_v2.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v2 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v2 bottleneck block.
  """
  return Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])
resnet_v2.default_image_size = 224


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_50.default_image_size = resnet_v2.default_image_size


def resnet_v2_101(inputs,
                  global_pool=True,
                  output_stride=None,
                  mixup_features=dict({}),
                  method=None,
                  num_classes=None,
                  is_training=True,
                  final_endpoint='Mixed_7c',
                  mixup_keep_prob=[0.8]*55,
                  random_keep_prob=[0.8]*55,
                  spatial_squeeze=True,
                  reuse=None,
                  create_aux_logits=True,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes=num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope,
                   mixup_features=mixup_features, method=method, 
                   mixup_keep_prob=mixup_keep_prob, random_keep_prob=random_keep_prob)
resnet_v2_101.default_image_size = resnet_v2.default_image_size


def resnet_v2_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_152.default_image_size = resnet_v2.default_image_size


def resnet_v2_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v2_200.default_image_size = resnet_v2.default_image_size
