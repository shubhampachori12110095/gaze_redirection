"""
    This script contains all needed neural network layers.
"""
from __future__ import division

import tensorflow as tf
import numpy as np


weight_init = tf.contrib.layers.xavier_initializer()


def batch_norm(input_):
    """Batch normalization based on tf.contrib.layers.

    """
    return tf.layers.batch_normalization(input_)


def instance_norm(x, scope='instance_norm'):
    return tf.contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def conv2d(input_, output_dim, d_h=2, d_w=2, scope='conv_0',
           conv_filters_dim=4, padding='zero', use_bias=True, pad=0):
    """Convolutional layer.

    Args:
        input_: should be a 4d tensor with [num_points, dim1, dim2, dim3].
        output_dim: number of channels of outputs.
        d_h: height of stride.
        d_w: width of stride.
        scope: name of variable scope.
        conv_filters_dim: size of kernel, width = height.
        padding: strategy of padding.
        use_bias: whether to use bias in this layer.
        pad: size of padding.

    """

    k_initializer = tf.random_normal_initializer(stddev=0.02)
    b_initializer = tf.constant_initializer(0)
    k_h = k_w = conv_filters_dim

    with tf.variable_scope(scope):

        if padding == 'zero':
            x = tf.pad(input_, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        elif padding == 'reflect':
            x = tf.pad(input_, [[0, 0], [pad, pad], [pad, pad], [0, 0]],
                       mode='REFLECT')
        else:
            x = input_

        conv = tf.layers.conv2d(x, output_dim, kernel_size=[k_h, k_w],
                                strides=(d_h, d_w),
                                kernel_initializer=k_initializer,
                                bias_initializer=b_initializer,
                                use_bias=use_bias)

    return conv


def conv2d_bn_relu(name, input_, filters, kernel_size, stride=1,
                   padding='VALID', act_relu=True, bn=True):

    with tf.variable_scope(name):

        conv = tf.layers.conv2d(
            input_,
            filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            kernel_initializer=tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            name='conv',
            data_format='channels_last'
        )

        if bn:
            conv = tf.layers.batch_normalization(
                conv, name='bn', momentum=0.1, epsilon=1e-5,
                training=True, trainable=True,
                gamma_initializer=tf.random_uniform_initializer(0, 1))

        if act_relu:
            conv = tf.nn.relu(conv)

    return conv


def deconv2d(input_, output_dim, d_h=2, d_w=2, scope='deconv_0',
             conv_filters_dim=4, padding='SAME', use_bias=True):
    """Transposed convolution (fractional stride convolution) layer.

    """

    k_initializer = tf.random_normal_initializer(stddev=0.02)
    b_initializer = tf.constant_initializer(0)
    k_h = k_w = conv_filters_dim

    with tf.variable_scope(scope):

        deconv = tf.layers.conv2d_transpose(
            input_, output_dim, kernel_size=[k_h, k_w],
            strides=(d_h, d_w), padding=padding,
            kernel_initializer=k_initializer, bias_initializer=b_initializer,
            use_bias=use_bias)

    return deconv


def skip_layer(input_, output_dim, scope):

    if int(input_.shape[3]) == output_dim:

        return tf.identity(input_, name='identity')

    else:

        return conv2d_bn_relu(scope, input_, filters=output_dim, kernel_size=1,
                              act_relu=False, bn=False)


def conv_block(input_, output_dim, scope):

    with tf.variable_scope(scope):

        bn = tf.layers.batch_normalization(
            input_, name='bn', momentum=0.1, epsilon=1e-5,
            training=True, trainable=True,
            gamma_initializer=tf.random_uniform_initializer(0, 1))
        act = relu(bn)

        conv_1 = conv2d_bn_relu('conv_1', act, filters=output_dim / 2,
                                kernel_size=1)
        conv_2 = conv2d_bn_relu('conv_2', conv_1, filters=output_dim / 2,
                                kernel_size=3, padding='SAME')
        conv_3 = conv2d_bn_relu('conv_3', conv_2, filters=output_dim,
                                kernel_size=1, padding='SAME', act_relu=False,
                                bn=False)

    return conv_3


def residual(input_, output_dim, scope):

    with tf.variable_scope(scope):

        conv = conv_block(input_, output_dim, 'conv_b')
        skip = skip_layer(conv, output_dim, 'skip')
        conv = tf.add_n([conv, skip])

    return conv


def max_pool(input_, kernel_size=2, stride=2, padding='VALID'):

    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def hourglass(input_, n, output_dim, n_modules, scope=None):

    with tf.variable_scope(scope):

        up1 = input_
        for i in range(n_modules):
            up1 = residual(up1, output_dim, scope='up1_%d' % i)

        low1 = max_pool(input_, kernel_size=2, stride=2)
        for i in range(n_modules):
            low1 = residual(low1, output_dim, scope='low1_%d' % i)

        if n > 1:
            low2 = hourglass(low1, n - 1, output_dim, n_modules, scope='low2')
        else:
            low2 = low1
            for i in range(n_modules):
                low2 = residual(low2, output_dim, scope='low2_%d' % i)
        low3 = low2
        for i in range(n_modules):
            low3 = residual(low3, output_dim, scope='low3_%d' % i)

        up2 = tf.image.resize_nearest_neighbor(low3, [int(low3.shape[1]) * 2,
                                                      int(low3.shape[2]) * 2])
    return tf.add_n([up1, up2])


def relu(input_):
    return tf.nn.relu(input_)


def lrelu(input_):
    return tf.nn.leaky_relu(input_, alpha=0.01)


def tanh(input_):
    return tf.tanh(input_)


def l1_loss(x, y):

    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def l2_loss(x, y):

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=[1, 2, 3]))

    return loss


def content_loss(endpoints_mixed, content_layers):

    loss = 0
    for layer in content_layers:
        feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
        size = tf.size(feat_a)
        loss += tf.nn.l2_loss(feat_a - feat_b) * 2 / tf.to_float(size)

    return loss


def style_loss(endpoints_mixed, style_layers):

    loss = 0
    for layer in style_layers:
        feat_a, feat_b = tf.split(endpoints_mixed[layer], 2, 0)
        size = tf.size(feat_a)
        loss += tf.nn.l2_loss(
            gram(feat_a) - gram(feat_b)) * 2 / tf.to_float(size)

    return loss


def gram(layer):

    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    features = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    denominator = tf.to_float(width * height * num_filters)
    grams = tf.matmul(features, features, transpose_a=True) / denominator

    return grams


def spher2cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    res = np.stack([x, y, z], axis=1)

    return res


def angular_dist(angle):
    angle = angle / 180. * np.pi

    theta = angle[:, 0]
    phi = angle[:, 1]

    coord_1 = spher2cart(theta, np.zeros(theta.shape))
    coord_2 = spher2cart(phi, np.pi / 2. * np.ones(phi.shape))

    dist = np.linalg.norm(coord_1 - coord_2, axis=1)

    angle = np.arcsin(dist / 2.) * 2

    return angle * 180. / np.pi


def angular2cart(angular):

    """
    :param angular: [yaw, pitch]
    :return: coordinates in cartesian system.
    """

    theta = angular[:, 0] / 180.0 * np.pi
    phi = angular[:, 1] / 180.0 * np.pi
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    return np.stack([x, y, z], axis=1)


def angular_error(x, y):
    x = angular2cart(x)
    y = angular2cart(y)

    x_norm = np.sqrt(np.sum(np.square(x), axis=1))
    y_norm = np.sqrt(np.sum(np.square(y), axis=1))

    sim = np.divide(np.sum(np.multiply(x, y), axis=1),
                    np.multiply(x_norm, y_norm))

    sim = np.clip(sim, -1.0 + 1e-6, 1.0 - 1e-6)

    return np.arccos(sim) * 180.0 / np.pi


if __name__ == '__main__':

    x = np.array([
        np.array([-15, 10]),
        np.array([5, 10])
    ])

    y = np.array([
        np.array([15, -10]),
        np.array([0, 15])
    ])

    delta = angular_error(x, y)

    print(delta)
