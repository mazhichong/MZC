import numpy as np
import tensorflow as tf
from utils import *
from activation import taylor_softmax
from tensorflow.keras.layers import Layer


reg = tf.contrib.layers.l2_regularizer(1.5e-4)
conv2d=tf.layers.conv2d
class IDRNet(object):
    def __init__(self, reg_rate=reg,  activate=tf.nn.elu,norm=tf.contrib.layers.instance_norm):
        self.reg_rate = reg_rate
        self.activate = activate
        self.norm = norm

    def forward(self, input_x1, layer=50):
        if layer == 38:
            layers = [3, 3, 3, 3]
        if layer == 50:
            layers = [3, 4, 6, 3]

        input_x = self.Conv2D(input_x1, 1, filters=64)
        out1 = self.block(input_x, layers[0])
        out1_damp = self.downsample(out1, 1, filters=128)
        out2 = self.block(out1_damp, layers[1], 128)
        out3 = self.block(out2, layers[2], 128)
        out4 = self.block(out3, layers[3], 128)
        return out4

    def block(self, inputx, layers, out_channels=64):
        d = 1
        x = inputx
        for i in range(layers):
            out1 = self.subblock(x, channels=out_channels, dilations=d)
            x = tf.concat(values=[x, out1], axis=-1)
            d = 2 * d
            if d > 16:
                d = 1
        out = self.Conv2D(x, 1, filters=out_channels)
        out += inputx
        return out

    def subblock(self, input_x, channels=64, dilations=1):
        x = tf.layers.separable_conv2d(input_x,channels,kernel_size=7,dilation_rate=dilations,padding="same", depthwise_regularizer=self.reg_rate,pointwise_regularizer=self.reg_rate)
        x = self.norm(x)
        x = tf.layers.conv2d(x, kernel_size=1, filters=4 * channels, dilation_rate=dilations, padding="same",
                             kernel_regularizer=self.reg_rate)
        x = self.activate(x)
        out = tf.layers.conv2d(x, kernel_size=1, filters=channels, dilation_rate=dilations, padding="same",
                               kernel_regularizer=self.reg_rate)

        return out

    def Conv2D(self, x, kernel_size=1, dilation=1, filters=64, padding='same', normalize=True, activation=True):
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, dilation_rate=dilation, padding=padding,
                             kernel_regularizer=self.reg_rate)
        if normalize:
            x = self.norm(x)
        if activation:
            x = self.activate(x)

        return x
    def downsample(self, x, kernel_size=2, dilation=1, filters=64, padding='same', normalize=True, activation=True):

        if normalize:
            x = self.norm(x)
        if activation:
            x = self.activate(x)
        x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, dilation_rate=dilation, padding=padding,
                             kernel_regularizer=self.reg_rate)  # 卷积

        return x


#Extract 1D-2D features
class PSSM(Layer):
  # modified from MRF to only output tiled 1D features
  def __init__(self, diag=0.4, use_entropy=False):
    super(PSSM, self).__init__()
    self.diag = diag
    self.use_entropy = use_entropy
  def call(self, inputs):
    x,y = inputs
    _,L,A = [tf.shape(y)[k] for k in range(3)]
    with tf.name_scope('1d_features'):
      # sequence
      x_i = x[0,:,:20]
      # pssm
      f_i = y[0]
      # entropy
      h_i = tf.zeros((L,1))
      # tile and combined 1D features
      feat_1D = tf.concat([x_i,f_i,h_i], axis=-1)
      feat_1D_tile_A = tf.tile(feat_1D[:,None,:], [1,L,1])
      feat_1D_tile_B = tf.tile(feat_1D[None,:,:], [L,1,1])

    with tf.name_scope('2d_features'):
      ic = self.diag * tf.eye(L*A)
      ic = tf.reshape(ic,(L,A,L,A))
      ic = tf.transpose(ic,(0,2,1,3))
      ic = tf.reshape(ic,(L,L,A*A))
      i0 = tf.zeros([L,L,1])
      feat_2D = tf.concat([ic,i0], axis=-1)

    feat = tf.concat([feat_1D_tile_A, feat_1D_tile_B, feat_2D],axis=-1)
    return tf.reshape(feat, [1,L,L,442+2*42])

#Predict residue distance and orientations using Taylor_Softmax
class Taylor_Softmax:
    def __init__(self, fliters=1, kernel_size=1, strides=(1, 1), padding='SAME', reg_rate=1.5e-4):
        self.reg_rate = reg_rate
        self.fliters = fliters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def forward(self, x):
        prob_theta = taylor_softmax(tf.layers.conv2d(x, 25 * self.fliters, self.kernel_size, self.strides, self.padding,
                                           kernel_regularizer=self.reg_rate))
        prob_phi = taylor_softmax(tf.layers.conv2d(x, 13 * self.fliters, self.kernel_size, self.strides, self.padding,
                                         kernel_regularizer=self.reg_rate))
        x = x + tf.transpose(x, perm=[0, 2, 1, 3])
        prob_dist = taylor_softmax(tf.layers.conv2d(x, 37 * self.fliters, self.kernel_size, self.strides, self.padding,
                                          kernel_regularizer=self.reg_rate))
        prob_omega = taylor_softmax(tf.layers.conv2d(x, 25 * self.fliters, self.kernel_size, self.strides, self.padding,
                                           kernel_regularizer=self.reg_rate))
        return prob_theta, prob_phi, prob_dist, prob_omega