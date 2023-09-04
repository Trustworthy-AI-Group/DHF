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
"""Contains the definition of the Inception Resnet V2 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim


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


dhf_layers = "80_81_82_84_90_91_92_94_100_101_102_104_110_111_112_114_118_119_120_124_125_126_128_132_133_134_136_140_141_142_144_148_149_150_152_156_157_158_160_164_165_166_168_172_173_174_176_180_181_182_184_188_189_190_192_196_197_198_200_204_205_206_208_212_213_214_216_220_221_222_224_228_229_230_232_236_237_238_240_244_245_246_248_252_253_254_256_260_261_262_264_268_269_270_272_276_277_278_280_288_289_290_294_295_296_298_302_303_304_306_310_311_312_314_318_319_320_322_326_327_328_330_334_335_336_338_342_343_344_346_350_351_352_354_358_359_360_362_366_367_368_370"


layer_idx_to_name_map = {
  1: 'Conv2d_1a_3x3', # 1 
  2: 'Conv2d_2a_3x3', # 2
  3: 'Conv2d_2b_3x3', # 3
  4: 'MaxPool_3a_3x3',  # 4 
  5: 'Conv2d_3b_1x1', # 5
  6: 'Conv2d_4a_3x3', # 6
  7: 'MaxPool_5a_3x3',  # 7
  8: 'Mixed_5b_B0_conv1_8', # 
  9: 'Mixed_5b_B1_conv1_9', #
  10: 'Mixed_5b_B1_conv5_10', # 
  11: 'Mixed_5b_B2_conv1_11', # 8
  12: 'Mixed_5b_B2_conv3_12', # 9
  13: 'Mixed_5b_B2_conv3_13', # 10
  14: 'Mixed_5b_B3_avgpool_14', # 
  15: 'Mixed_5b_B3_conv1_15', #
  16: 'Mixed_5b', # 
  17: 'B0_conv1_17',  # 
  18: 'B1_conv1_18',  # 
  19: 'B1_conv3_19',  # 
  20: 'B2_conv1_20',  # 11
  21: 'B2_conv3_21',  # 12
  22: 'B2_conv3_22',  # 13
  23: 'concated_23',  # 
  24: 'up_conv1_24',  # 14
  25: 'shortcut_25',  # 
  26: 'output_26',  # 
  27: 'B0_conv1_27',  # 
  28: 'B1_conv1_28',  # 
  29: 'B1_conv3_29',  # 
  30: 'B2_conv1_30',  # 15
  31: 'B2_conv3_31',  # 16
  32: 'B2_conv3_32',  # 17
  33: 'concated_33',  # 
  34: 'up_conv1_34',  # 18
  35: 'shortcut_35',  # 
  36: 'output_36',  # 
  37: 'B0_conv1_37',  # 
  38: 'B1_conv1_38',  # 
  39: 'B1_conv3_39',  # 
  40: 'B2_conv1_40',  # 19
  41: 'B2_conv3_41',  # 20
  42: 'B2_conv3_42',  # 21
  43: 'concated_43',  # 
  44: 'up_conv1_44',  # 22
  45: 'shortcut_45',  # 
  46: 'output_46',  # 
  47: 'B0_conv1_47',  # 
  48: 'B1_conv1_48',  # 
  49: 'B1_conv3_49',  # 
  50: 'B2_conv1_50',  # 23
  51: 'B2_conv3_51',  # 24
  52: 'B2_conv3_52',  # 25
  53: 'concated_53',  # 
  54: 'up_conv1_54',  # 26
  55: 'shortcut_55',  # 
  56: 'output_56',  # 
  57: 'B0_conv1_57',  # 
  58: 'B1_conv1_58',  # 
  59: 'B1_conv3_59',  # 
  60: 'B2_conv1_60',  # 27
  61: 'B2_conv3_61',  # 28
  62: 'B2_conv3_62',  # 29
  63: 'concated_63',  # 
  64: 'up_conv1_64',  # 30
  65: 'shortcut_65',  # 
  66: 'output_66',  # 
  67: 'B0_conv1_67',  # 
  68: 'B1_conv1_68',  # 
  69: 'B1_conv3_69',  # 
  70: 'B2_conv1_70',  # 31
  71: 'B2_conv3_71',  # 32
  72: 'B2_conv3_72',  # 33
  73: 'concated_73',  # 
  74: 'up_conv1_74',  # 34
  75: 'shortcut_75',  # 
  76: 'output_76',  # 
  77: 'B0_conv1_77',  # 
  78: 'B1_conv1_78',  # 
  79: 'B1_conv3_79',  # 
  80: 'B2_conv1_80',  # 35
  81: 'B2_conv3_81',  # 36
  82: 'B2_conv3_82',  # 37
  83: 'concated_83',  # 
  84: 'up_conv1_84',  # 38
  85: 'shortcut_85',  # 
  86: 'output_86',  # 
  87: 'B0_conv1_87',  # 
  88: 'B1_conv1_88',  # 
  89: 'B1_conv3_89',  # 
  90: 'B2_conv1_90',  # 39
  91: 'B2_conv3_91',  # 40
  92: 'B2_conv3_92',  # 41
  93: 'concated_93',  # 
  94: 'up_conv1_94',  # 42
  95: 'shortcut_95',  # 
  96: 'output_96',  # 
  97: 'B0_conv1_97',  # 
  98: 'B1_conv1_98',  # 
  99: 'B1_conv3_99',  # 
  100: 'B2_conv1_100',  # 43
  101: 'B2_conv3_101',  # 44
  102: 'B2_conv3_102',  # 45
  103: 'concated_103',  # 
  104: 'up_conv1_104',  # 46
  105: 'shortcut_105',  #
  106: 'output_106',  # 
  107: 'B0_conv1_107',  # 
  108: 'B1_conv1_108',  # 
  109: 'B1_conv3_109',  # 
  110: 'B2_conv1_110',  # 47
  111: 'B2_conv3_111',  # 48
  112: 'B2_conv3_112',  # 49
  113: 'concated_113',  # 
  114: 'up_conv1_114',  # 50
  115: 'shortcut_115',  # 
  116: 'output_116',  # 
  117: 'Mixed_6a_B0_conv3_117', # 
  118: 'Mixed_6a_B1_conv1_118', # 51
  119: 'Mixed_6a_B1_conv3_119', # 52
  120: 'Mixed_6a_B1_conv3_120', # 53
  121: 'Mixed_6a_B2_maxpool_121', # 
  122: 'Mixed_6a',  # 
  123: 'B0_conv1_123',  # 
  124: 'B1_conv1_124',  # 54
  125: 'B1_conv1x7_125',  # 55
  126: 'B1_conv7x1_126',  # 56
  127: 'concated_127',  # 
  128: 'up_conv1_128',  # 57
  129: 'shortcut_129',  # 
  130: 'output_130',  # 
  131: 'B0_conv1_131',  # 
  132: 'B1_conv1_132',  # 58
  133: 'B1_conv1x7_133',  # 59
  134: 'B1_conv7x1_134',  # 60
  135: 'concated_135',  # 
  136: 'up_conv1_136',  # 61
  137: 'shortcut_137',  # 
  138: 'output_138',  # 
  139: 'B0_conv1_139',  # 
  140: 'B1_conv1_140',  # 62
  141: 'B1_conv1x7_141',  # 63
  142: 'B1_conv7x1_142',  # 64
  143: 'concated_143',  # 
  144: 'up_conv1_144',  # 65
  145: 'shortcut_145',  # 
  146: 'output_146',  # 
  147: 'B0_conv1_147',  # 
  148: 'B1_conv1_148',  # 66
  149: 'B1_conv1x7_149',  # 67
  150: 'B1_conv7x1_150',  # 68
  151: 'concated_151',  # 
  152: 'up_conv1_152',  # 69
  153: 'shortcut_153',  # 
  154: 'output_154',  # 
  155: 'B0_conv1_155',  # 
  156: 'B1_conv1_156',  # 70
  157: 'B1_conv1x7_157',  # 71
  158: 'B1_conv7x1_158',  # 72
  159: 'concated_159',  # 
  160: 'up_conv1_160',  # 73
  161: 'shortcut_161',  # 
  162: 'output_162',  # 
  163: 'B0_conv1_163',  # 
  164: 'B1_conv1_164',  # 74
  165: 'B1_conv1x7_165',  # 75
  166: 'B1_conv7x1_166',  # 76
  167: 'concated_167',  # 
  168: 'up_conv1_168',  # 77
  169: 'shortcut_169',  # 
  170: 'output_170',  # 
  171: 'B0_conv1_171',  # 
  172: 'B1_conv1_172',  # 78
  173: 'B1_conv1x7_173',  # 79
  174: 'B1_conv7x1_174',  # 80
  175: 'concated_175',  # 
  176: 'up_conv1_176',  # 81
  177: 'shortcut_177',  # 
  178: 'output_178',  # 
  179: 'B0_conv1_179',  # 
  180: 'B1_conv1_180',  # 82
  181: 'B1_conv1x7_181',  # 83
  182: 'B1_conv7x1_182',  # 84
  183: 'concated_183',  # 
  184: 'up_conv1_184',  # 85
  185: 'shortcut_185',  # 
  186: 'output_186',  # 
  187: 'B0_conv1_187',  # 
  188: 'B1_conv1_188',  # 86
  189: 'B1_conv1x7_189',  # 87
  190: 'B1_conv7x1_190',  # 88
  191: 'concated_191',  # 
  192: 'up_conv1_192',  # 89
  193: 'shortcut_193',  # 
  194: 'output_194',  # 
  195: 'B0_conv1_195',  # 
  196: 'B1_conv1_196',  # 90
  197: 'B1_conv1x7_197',  # 91
  198: 'B1_conv7x1_198',  # 92
  199: 'concated_199',  # 
  200: 'up_conv1_200',  # 93
  201: 'shortcut_201',  # 
  202: 'output_202',  # 
  203: 'B0_conv1_203',  # 
  204: 'B1_conv1_204',  # 94
  205: 'B1_conv1x7_205',  # 95
  206: 'B1_conv7x1_206',  # 96
  207: 'concated_207',  # 
  208: 'up_conv1_208',  # 97
  209: 'shortcut_209',  # 
  210: 'output_210',  # 
  211: 'B0_conv1_211',  # 
  212: 'B1_conv1_212',  # 98
  213: 'B1_conv1x7_213',  # 99
  214: 'B1_conv7x1_214',  # 100
  215: 'concated_215',  # 
  216: 'up_conv1_216',  # 101
  217: 'shortcut_217',  # 
  218: 'output_218',  # 
  219: 'B0_conv1_219',  # 
  220: 'B1_conv1_220',  # 102
  221: 'B1_conv1x7_221',  # 103
  222: 'B1_conv7x1_222',  # 104
  223: 'concated_223',  # 
  224: 'up_conv1_224',  # 105
  225: 'shortcut_225',  # 
  226: 'output_226',  # 
  227: 'B0_conv1_227',  # 
  228: 'B1_conv1_228',  # 106
  229: 'B1_conv1x7_229',  # 107
  230: 'B1_conv7x1_230',  # 108
  231: 'concated_231',  # 
  232: 'up_conv1_232',  # 109
  233: 'shortcut_233',  # 
  234: 'output_234',  # 
  235: 'B0_conv1_235',  # 
  236: 'B1_conv1_236',  # 110
  237: 'B1_conv1x7_237',  # 111
  238: 'B1_conv7x1_238',  # 112
  239: 'concated_239',  # 
  240: 'up_conv1_240',  # 113
  241: 'shortcut_241',  # 
  242: 'output_242',  # 
  243: 'B0_conv1_243',  # 
  244: 'B1_conv1_244',  # 114
  245: 'B1_conv1x7_245',  # 115
  246: 'B1_conv7x1_246',  # 116
  247: 'concated_247',  # 
  248: 'up_conv1_248',  # 117
  249: 'shortcut_249',  # 
  250: 'output_250',  # 
  251: 'B0_conv1_251',  # 
  252: 'B1_conv1_252',  # 118
  253: 'B1_conv1x7_253',  # 119
  254: 'B1_conv7x1_254',  # 120
  255: 'concated_255',  # 
  256: 'up_conv1_256',  # 121
  257: 'shortcut_257',  # 
  258: 'output_258',  # 
  259: 'B0_conv1_259',  # 
  260: 'B1_conv1_260',  # 122
  261: 'B1_conv1x7_261',  # 123
  262: 'B1_conv7x1_262',  # 124
  263: 'concated_263',  # 
  264: 'up_conv1_264',  # 125
  265: 'shortcut_265',  # 
  266: 'output_266',  # 
  267: 'B0_conv1_267',  # 
  268: 'B1_conv1_268',  # 126
  269: 'B1_conv1x7_269',  # 127
  270: 'B1_conv7x1_270',  # 128
  271: 'concated_271',  # 
  272: 'up_conv1_272',  # 129
  273: 'shortcut_273',  # 
  274: 'output_274',  # 
  275: 'B0_conv1_275',  # 
  276: 'B1_conv1_276',  # 130
  277: 'B1_conv1x7_277',  # 131
  278: 'B1_conv7x1_278',  # 132
  279: 'concated_279',  # 
  280: 'up_conv1_280',  # 133
  281: 'shortcut_281',  # 
  282: 'output_282',  # 
  283: 'PreAuxLogits',  # 
  284: 'Mixed_7a_B0_conv1_284', # 
  285: 'Mixed_7a_B0_conv3_285', # 
  286: 'Mixed_7a_B1_conv1_286', # 
  287: 'Mixed_7a_B1_conv3_287', # 
  288: 'Mixed_7a_B2_conv1_288', # 134
  289: 'Mixed_7a_B2_conv3_289', # 135
  290: 'Mixed_7a_B2_conv3_290', # 136
  291: 'Mixed_7a_B3_maxpool_291', # 
  292: 'Mixed_7a',  # 
  293: 'B0_conv1_293',  # 
  294: 'B1_conv1_294',  # 137
  295: 'B1_conv1x3_295',  # 138
  296: 'B1_conv3x1_296',  # 139
  297: 'concated_297',  # 
  298: 'up_conv1_298',  # 140
  299: 'shortcut_299',  # 
  300: 'output_300',  # 
  301: 'B0_conv1_301',  # 
  302: 'B1_conv1_302',  # 141
  303: 'B1_conv1x3_303',  # 142
  304: 'B1_conv3x1_304',  # 143
  305: 'concated_305',  # 
  306: 'up_conv1_306',  # 144
  307: 'shortcut_307',  # 
  308: 'output_308',  # 
  309: 'B0_conv1_309',  # 
  310: 'B1_conv1_310',  # 145
  311: 'B1_conv1x3_311',  # 146
  312: 'B1_conv3x1_312',  # 147
  313: 'concated_313',  # 
  314: 'up_conv1_314',  # 148
  315: 'shortcut_315',  # 
  316: 'output_316',  # 
  317: 'B0_conv1_317',  # 
  318: 'B1_conv1_318',  # 149
  319: 'B1_conv1x3_319',  # 150
  320: 'B1_conv3x1_320',  # 151
  321: 'concated_321',  # 
  322: 'up_conv1_322',  # 152
  323: 'shortcut_323',  # 
  324: 'output_324',  # 
  325: 'B0_conv1_325',  # 
  326: 'B1_conv1_326',  # 153
  327: 'B1_conv1x3_327',  # 154
  328: 'B1_conv3x1_328',  # 155
  329: 'concated_329',  # 
  330: 'up_conv1_330',  # 156
  331: 'shortcut_331',  # 
  332: 'output_332',  # 
  333: 'B0_conv1_333',  # 
  334: 'B1_conv1_334',  # 157
  335: 'B1_conv1x3_335',  # 158
  336: 'B1_conv3x1_336',  # 159
  337: 'concated_337',  # 
  338: 'up_conv1_338',  # 160
  339: 'shortcut_339',  # 
  340: 'output_340',  # 
  341: 'B0_conv1_341',  # 
  342: 'B1_conv1_342',  # 161
  343: 'B1_conv1x3_343',  # 162
  344: 'B1_conv3x1_344',  # 163
  345: 'concated_345',  # 
  346: 'up_conv1_346',  # 164
  347: 'shortcut_347',  # 
  348: 'output_348',  # 
  349: 'B0_conv1_349',  # 
  350: 'B1_conv1_350',  # 165
  351: 'B1_conv1x3_351',  # 166
  352: 'B1_conv3x1_352',  # 167
  353: 'concated_353',  # 
  354: 'up_conv1_354',  # 168
  355: 'shortcut_355',  # 
  356: 'output_356',  # 
  357: 'B0_conv1_357',  # 
  358: 'B1_conv1_358',  # 169
  359: 'B1_conv1x3_359',  # 170
  360: 'B1_conv3x1_360',  # 171
  361: 'concated_361',  # 
  362: 'up_conv1_362',  # 172
  363: 'shortcut_363',  # 
  364: 'output_364',  # 
  365: 'B0_conv1_365',  # 
  366: 'B1_conv1_366',  # 173
  367: 'B1_conv1x3_367',  # 174
  368: 'B1_conv3x1_368',  # 175
  369: 'concated_369',  # 
  370: 'up_conv1_370',  # 176
  371: 'shortcut_371',  # 
  372: 'output_372',  # 
  373: 'Conv2d_7b_1x1', # 
  374: 'PreLogitsFlatten',  # 
  375: 'Logits',  # 
  376: 'Predictions' # 
  # 6 conv
  # 1 maxpool   # 7 layers

  # 1 mixed_5b
  # 10 block35
  # 1 block_6a
  # 20 block17
  # 1 mixed_7a
  # 9 block8
  # 1 block8   # 43 blocks in total

  # 1 conv     # 1 layer
}


g_layer_idx = 1


def block35(net, end_points, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None, 
            method="original", mixup_features=dict({}),
            mixup_keep_prob=tf.fill([1, 1], value=0.8), random_keep_prob=tf.fill([1, 1], value=0.8)):
  global g_layer_idx

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

  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
      end_points["B0_conv1_"+str(g_layer_idx)] = tower_conv
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv = our_method(tower_conv)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      end_points["B1_conv1_"+str(g_layer_idx)] = tower_conv1_0
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_0 = our_method(tower_conv1_0)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
      end_points["B1_conv3_"+str(g_layer_idx)] = tower_conv1_1
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_1 = our_method(tower_conv1_1)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      end_points["B2_conv1_"+str(g_layer_idx)] = tower_conv2_0
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv2_0 = our_method(tower_conv2_0)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      end_points["B2_conv3_"+str(g_layer_idx)] = tower_conv2_1
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv2_1 = our_method(tower_conv2_1)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
      end_points["B2_conv3_"+str(g_layer_idx)] = tower_conv2_2
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv2_2 = our_method(tower_conv2_2)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    end_points["concated_"+str(g_layer_idx)] = mixed
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      mixed = our_method(mixed)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    end_points["up_conv1_"+str(g_layer_idx)] = up
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      up = our_method(up)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
        # Use clip_by_value to simulate bandpass activation.
        scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    assert net.get_shape()[3] == 320

    end_points["shortcut_"+str(g_layer_idx)] = net
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      net = our_method(net)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)

    end_points["output_"+str(g_layer_idx)] = net
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      net = our_method(net)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1
  return net


def block17(net, end_points, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None,
            method="original", mixup_features=dict({}),
            mixup_keep_prob=tf.fill([1, 1], value=0.8), random_keep_prob=tf.fill([1, 1], value=0.8)):
  global g_layer_idx

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

  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
      end_points["B0_conv1_"+str(g_layer_idx)] = tower_conv
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv = our_method(tower_conv)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      end_points["B1_conv1_"+str(g_layer_idx)] = tower_conv1_0
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_0 = our_method(tower_conv1_0)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      end_points["B1_conv1x7_"+str(g_layer_idx)] = tower_conv1_1
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_1 = our_method(tower_conv1_1)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
      end_points["B1_conv7x1_"+str(g_layer_idx)] = tower_conv1_2
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_2 = our_method(tower_conv1_2)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    end_points["concated_"+str(g_layer_idx)] = mixed
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      mixed = our_method(mixed)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    end_points["up_conv1_"+str(g_layer_idx)] = up
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      up = our_method(up)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    assert net.get_shape()[3] == 1088

    end_points["shortcut_"+str(g_layer_idx)] = net
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      net = our_method(net)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    net += scaled_up
 
    if activation_fn:
      net = activation_fn(net)
    end_points["output_"+str(g_layer_idx)] = net
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      net = our_method(net)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1
  return net


def block8(net, end_points, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None,
           method="original", mixup_features=dict({}),
           mixup_keep_prob=tf.fill([1, 1], value=0.8), random_keep_prob=tf.fill([1, 1], value=0.8)):
  global g_layer_idx

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

  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
      end_points["B0_conv1_"+str(g_layer_idx)] = tower_conv
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv = our_method(tower_conv)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      end_points["B1_conv1_"+str(g_layer_idx)] = tower_conv1_0
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_0 = our_method(tower_conv1_0)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      end_points["B1_conv1x3_"+str(g_layer_idx)] = tower_conv1_1
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_1 = our_method(tower_conv1_1)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
      end_points["B1_conv3x1_"+str(g_layer_idx)] = tower_conv1_2
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        tower_conv1_2 = our_method(tower_conv1_2)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    end_points["concated_"+str(g_layer_idx)] = mixed
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      mixed = our_method(mixed)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    end_points["up_conv1_"+str(g_layer_idx)] = up
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      up = our_method(up)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    assert net.get_shape()[3] == 2080

    end_points["shortcut_"+str(g_layer_idx)] = net
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      net = our_method(net)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1

    net += scaled_up
 
    if activation_fn:
      net = activation_fn(net)
    end_points["output_"+str(g_layer_idx)] = net
    if method == "dhf" and g_layer_idx in mixup_features.keys():
      net = our_method(net)
      print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
    g_layer_idx += 1
  return net


def inception_resnet_v2_base(inputs,
                             mixup_features,
                             output_stride=16,
                             align_feature_maps=False,
                             method=None,
                             final_endpoint='Conv2d_7b_1x1', 
                             scope=None,
                             mixup_keep_prob=tf.fill([1, 1], value=0.8),
                             random_keep_prob=tf.fill([1, 1], value=0.8)):
  """Inception model from  http://arxiv.org/abs/1602.07261.

  Constructs an Inception Resnet v2 network from inputs to the given final
  endpoint. This method can construct the network up to the final inception
  block Conv2d_7b_1x1.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    output_stride: A scalar that specifies the requested ratio of input to
      output spatial resolution. Only supports 8 and 16.
    align_feature_maps: When true, changes all the VALID paddings in the network
      to SAME padding so that the feature maps are aligned.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
      or if the output_stride is not 8 or 16, or if the output_stride is 8 and
      we request an end point after 'PreAuxLogits'.
  """
  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  padding = 'SAME' if align_feature_maps else 'VALID'

  end_points = {}
  global g_layer_idx 
  g_layer_idx = 1

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

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 149 x 149 x 32
      net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                        scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
      g_layer_idx += 1

      # 147 x 147 x 32
      net = slim.conv2d(net, 32, 3, padding=padding,
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      g_layer_idx += 1
      # 147 x 147 x 64
      net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      g_layer_idx += 1
      # 73 x 73 x 64
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_3a_3x3')
      if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
      g_layer_idx += 1
      # 73 x 73 x 80
      net = slim.conv2d(net, 80, 1, padding=padding,
                        scope='Conv2d_3b_1x1')
      if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
      g_layer_idx += 1
      # 71 x 71 x 192
      net = slim.conv2d(net, 192, 3, padding=padding,
                        scope='Conv2d_4a_3x3')
      if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
      g_layer_idx += 1
      # 35 x 35 x 192
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_5a_3x3')
      if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points
      g_layer_idx += 1

      # 35 x 35 x 320
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
          end_points["Mixed_5b_B0_conv1_"+str(g_layer_idx)] = tower_conv
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv = our_method(tower_conv)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
          end_points["Mixed_5b_B1_conv1_"+str(g_layer_idx)] = tower_conv1_0
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv1_0 = our_method(tower_conv1_0)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                      scope='Conv2d_0b_5x5')
          end_points["Mixed_5b_B1_conv5_"+str(g_layer_idx)] = tower_conv1_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv1_1 = our_method(tower_conv1_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
          end_points["Mixed_5b_B2_conv1_"+str(g_layer_idx)] = tower_conv2_0
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv2_0 = our_method(tower_conv2_0)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                      scope='Conv2d_0b_3x3')
          end_points["Mixed_5b_B2_conv3_"+str(g_layer_idx)] = tower_conv2_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv2_1 = our_method(tower_conv2_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                      scope='Conv2d_0c_3x3')
          end_points["Mixed_5b_B2_conv3_"+str(g_layer_idx)] = tower_conv2_2
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv2_2 = our_method(tower_conv2_2)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                       scope='AvgPool_0a_3x3')
          end_points["Mixed_5b_B3_avgpool_"+str(g_layer_idx)] = tower_pool
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_pool = our_method(tower_pool)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                     scope='Conv2d_0b_1x1')
          end_points["Mixed_5b_B3_conv1_"+str(g_layer_idx)] = tower_pool_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_pool_1 = our_method(tower_pool_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        net = tf.concat(
            [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

      if add_and_check_final('Mixed_5b', net): return net, end_points
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      # TODO(alemi): Register intermediate endpoints
      net = slim.repeat(net, 10, block35, end_points=end_points, scale=0.17,
            method=method, mixup_features=mixup_features,
            mixup_keep_prob=mixup_keep_prob, random_keep_prob=random_keep_prob)

      # 17 x 17 x 1088 if output_stride == 8,
      # 33 x 33 x 1088 if output_stride == 16
      use_atrous = output_stride == 8

      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                   padding=padding,
                                   scope='Conv2d_1a_3x3')
          end_points["Mixed_6a_B0_conv3_"+str(g_layer_idx)] = tower_conv
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv = our_method(tower_conv)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          end_points["Mixed_6a_B1_conv1_"+str(g_layer_idx)] = tower_conv1_0
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv1_0 = our_method(tower_conv1_0)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                      scope='Conv2d_0b_3x3')
          end_points["Mixed_6a_B1_conv3_"+str(g_layer_idx)] = tower_conv1_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv1_1 = our_method(tower_conv1_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                      stride=1 if use_atrous else 2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
          end_points["Mixed_6a_B1_conv3_"+str(g_layer_idx)] = tower_conv1_2
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv1_2 = our_method(tower_conv1_2)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
          end_points["Mixed_6a_B2_maxpool_"+str(g_layer_idx)] = tower_pool
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_pool = our_method(tower_pool)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

      if add_and_check_final('Mixed_6a', net): return net, end_points
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      # TODO(alemi): register intermediate endpoints
      with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
        net = slim.repeat(net, 20, block17, end_points=end_points, scale=0.10,
                  method=method, mixup_features=mixup_features,
                  mixup_keep_prob=mixup_keep_prob, random_keep_prob=random_keep_prob)
      if add_and_check_final('PreAuxLogits', net): return net, end_points
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      if output_stride == 8:
        # TODO(gpapan): Properly support output_stride for the rest of the net.
        raise ValueError('output_stride==8 is only supported up to the '
                         'PreAuxlogits end_point for now.')

      # 8 x 8 x 2080
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          end_points["Mixed_7a_B0_conv1_"+str(g_layer_idx)] = tower_conv
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv = our_method(tower_conv)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                     padding=padding,
                                     scope='Conv2d_1a_3x3')
          end_points["Mixed_7a_B0_conv3_"+str(g_layer_idx)] = tower_conv_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv_1 = our_method(tower_conv_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          end_points["Mixed_7a_B1_conv1_"+str(g_layer_idx)] = tower_conv_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv_1 = our_method(tower_conv_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
          end_points["Mixed_7a_B1_conv3_"+str(g_layer_idx)] = tower_conv1_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv1_1 = our_method(tower_conv1_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          end_points["Mixed_7a_B2_conv1_"+str(g_layer_idx)] = tower_conv2
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv2 = our_method(tower_conv2)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                      scope='Conv2d_0b_3x3')
          end_points["Mixed_7a_B2_conv3_"+str(g_layer_idx)] = tower_conv2_1
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv2_1 = our_method(tower_conv2_1)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

          tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
          end_points["Mixed_7a_B2_conv3_"+str(g_layer_idx)] = tower_conv2_2
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_conv2_2 = our_method(tower_conv2_2)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool2d(net, 3, stride=2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
          end_points["Mixed_7a_B3_maxpool_"+str(g_layer_idx)] = tower_pool
          if method == "dhf" and g_layer_idx in mixup_features.keys():
            tower_pool = our_method(tower_pool)
            print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
          g_layer_idx += 1

        net = tf.concat(
            [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)

      if add_and_check_final('Mixed_7a', net): return net, end_points
      end_points[layer_idx_to_name_map[g_layer_idx]] = net
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1

      # TODO(alemi): register intermediate endpoints
      net = slim.repeat(net, 9, block8, end_points=end_points, scale=0.20,
                method=method, mixup_features=mixup_features,
                mixup_keep_prob=mixup_keep_prob, random_keep_prob=random_keep_prob)

      net = block8(net, end_points=end_points, activation_fn=None,
                   method=method, mixup_features=mixup_features,
                   mixup_keep_prob=mixup_keep_prob, random_keep_prob=random_keep_prob)

      # 8 x 8 x 1536
      net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
      if method == "dhf" and g_layer_idx in mixup_features.keys():
        net = our_method(net)
        print("Do feature mixup and tansform at layer-{} {}".format(g_layer_idx, layer_idx_to_name_map[g_layer_idx]))
      g_layer_idx += 1
      if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points

    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)


def inception_resnet_v2(inputs, 
                        mixup_features=dict({}),
                        method=None,
                        num_classes=1001,
                        is_training=True,
                        final_endpoint='Mixed_7c',
                        dropout_keep_prob=0.8,
                        mixup_keep_prob=tf.fill([1, 1], value=0.8),
                        random_keep_prob=tf.fill([1, 1], value=0.8),
                        reuse=None,
                        create_aux_logits=False,
                        scope='InceptionResnetV2',
                        ):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxilliary logits.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

      net, end_points = inception_resnet_v2_base(
          inputs, mixup_features, method=method, scope=scope,
          mixup_keep_prob=mixup_keep_prob, random_keep_prob=random_keep_prob)

      if create_aux_logits:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux

      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                              scope='AvgPool_1a_8x8')
        net = slim.flatten(net)

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')

        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    # print(end_points.keys())
    return logits, end_points
inception_resnet_v2.default_image_size = 299


def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001):
  """Returns the scope with the default parameters for inception_resnet_v2.

  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope
