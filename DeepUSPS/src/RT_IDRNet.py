import numpy as np
import tensorflow as tf
from utils import *
from activation import taylor_softmax

conv2d=tf.layers.conv2d
reg = tf.contrib.layers.l2_regularizer(1.5e-4)
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