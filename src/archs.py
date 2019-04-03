from __future__ import division
from utils.ops import *

import tensorflow as tf
import logging
import tensorflow.contrib.slim as slim


def discriminator(params, x_init, reuse=False):

    logging.info('************discriminator************')

    layers = 5
    channel = 64
    image_size = params.image_size

    with tf.variable_scope('discriminator', reuse=reuse):

        # 64 3 -> 32 64 -> 16 128 -> 8 256 -> 4 512 -> 2 1024

        x = conv2d(x_init, channel, conv_filters_dim=4, d_h=2, d_w=2,
                   scope='conv_0', pad=1, use_bias=True)
        x = lrelu(x)
        logging.info('%s_%d: %s' % (
            'discriminator_conv', 0, ','.join(map(str, x.get_shape().as_list()))))

        for i in range(1, layers):
            x = conv2d(x, channel * 2, conv_filters_dim=4, d_h=2, d_w=2,
                       scope='conv_%d' % i, pad=1, use_bias=True)
            x = lrelu(x)
            channel = channel * 2
            logging.info('%s_%d: %s' % (
                'discriminator_conv', i,
                ','.join(map(str, x.get_shape().as_list()))))

        filter_size = int(image_size / 2 ** layers)

        x_gan = conv2d(x, 1, conv_filters_dim=filter_size, d_h=1, d_w=1, pad=1,
                       scope='conv_logit_gan', use_bias=False)
        logging.info('%s_%s: %s' % (
            'discriminator_conv', 'gan',
            ','.join(map(str, x_gan.get_shape().as_list()))))

        x_reg = conv2d(x, 2, conv_filters_dim=filter_size,
                       d_h=1, d_w=1, scope='conv_logit_reg', use_bias=False)
        logging.info('%s_%s: %s' % (
            'discriminator_conv', 'reg',
            ','.join(map(str, x_reg.get_shape().as_list()))))
        x_reg = tf.reshape(x_reg, [-1, 2])

        logging.info('************discriminator************')
        return x_gan, x_reg


def generator(input_, angles, reuse=False):

    channel = 64
    style_dim = angles.get_shape().as_list()[-1]

    angles_reshaped = tf.reshape(angles, [-1, 1, 1, style_dim])
    angles_tiled = tf.tile(angles_reshaped, [1, tf.shape(input_)[1],
                                             tf.shape(input_)[2], 1])
    x = tf.concat([input_, angles_tiled], axis=3)

    with tf.variable_scope('generator', reuse=reuse):

        # input layer
        x = conv2d(x, channel, d_h=1, d_w=1, scope='conv2d_input',
                   use_bias=False, pad=3, conv_filters_dim=7)
        x = instance_norm(x, scope='in_input')
        x = relu(x)

        # encoder
        for i in range(2):

            x = conv2d(x, 2 * channel, d_h=2, d_w=2, scope='conv2d_%d' % i,
                       use_bias=False, pad=1, conv_filters_dim=4)
            x = instance_norm(x, scope='in_conv_%d' % i)
            x = relu(x)
            channel = 2 * channel
            logging.info('%s: %s' % (
                'generator', ','.join(map(str, x.get_shape().as_list()))))

        # bottleneck
        for i in range(6):

            x_a = conv2d(x, channel, conv_filters_dim=3, d_h=1, d_w=1,
                         pad=1, use_bias=False, scope='conv_res_a_%d' % i)
            x_a = instance_norm(x_a, 'in_res_a_%d' % i)
            x_a = relu(x_a)
            x_b = conv2d(x_a, channel, conv_filters_dim=3, d_h=1, d_w=1,
                         pad=1, use_bias=False, scope='conv_res_b_%d' % i)
            x_b = instance_norm(x_b, 'in_res_b_%d' % i)

            x = x + x_b

            logging.info('%s: %s' % (
                'generator', ','.join(map(str, x.get_shape().as_list()))))

        # decoder
        for i in range(2):

            x = deconv2d(x, int(channel / 2), conv_filters_dim=4, d_h=2, d_w=2,
                         use_bias=False, scope='deconv_%d' % i)
            x = instance_norm(x, scope='in_decon_%d' % i)
            x = relu(x)
            channel = int(channel / 2)
            logging.info('%s: %s' % (
                'generator', ','.join(map(str, x.get_shape().as_list()))))

        x = conv2d(x, 3, conv_filters_dim=7, d_h=1, d_w=1, pad=3,
                   use_bias=False, scope='output')
        x = tanh(x)
        logging.info('%s: %s' % (
            'generator', ','.join(map(str, x.get_shape().as_list()))))

    return x


def unet_generator(input_, angles, reuse=False):

    channel = 64
    angle_dim = angles.get_shape().as_list()[-1]

    angles_reshaped = tf.reshape(angles, [-1, 1, 1, angle_dim])
    angles_tiled = tf.tile(angles_reshaped, [1, tf.shape(input_)[1],
                                             tf.shape(input_)[2], 1])
    x = tf.concat([input_, angles_tiled], axis=3)

    features = []

    with tf.variable_scope('generator', reuse=reuse):

        # input layer
        x = conv2d(x, channel, d_h=1, d_w=1, scope='conv2d_input',
                   use_bias=False, pad=3, conv_filters_dim=7)
        x = instance_norm(x, scope='in_input')
        x = relu(x)
        logging.info('%s: %s' % (
            'generator', ','.join(map(str, x.get_shape().as_list()))))

        # encoder 128, 64 -> 64, 128 -> 32, 256 -> 16, 512 -> 8, 1024
        for i in range(4):

            features.append(x)
            x = conv2d(x, 2 * channel, d_h=2, d_w=2, scope='conv2d_%d' % i,
                       use_bias=False, pad=1, conv_filters_dim=4)
            x = instance_norm(x, scope='in_conv_%d' % i)
            x = relu(x)
            channel = 2 * channel
            logging.info('%s: %s' % (
                'generator', ','.join(map(str, x.get_shape().as_list()))))

        for i in range(4):

            x = deconv2d(x, int(channel / 2), conv_filters_dim=4, d_h=2, d_w=2,
                         use_bias=False, scope='deconv_%d' % i)
            x = instance_norm(x, scope='in_deconv_%d' % i)
            x = relu(x)
            channel = int(channel / 2)

            feature = features[4 - i - 1]
            x = tf.concat([x, feature], axis=-1)

            logging.info('%s: %s' % (
                'generator', ','.join(map(str, x.get_shape().as_list()))))

        x = conv2d(x, 3, conv_filters_dim=7, d_h=1, d_w=1, pad=3,
                   use_bias=False, scope='output')
        x = tanh(x)
        logging.info('%s: %s' % (
            'generator', ','.join(map(str, x.get_shape().as_list()))))

    return x


def predictor(input_, n_stacks=3, reuse=False):

    n_depth = 4
    n_features = 64
    n_modules = 1
    n_keypoints = 6

    with tf.variable_scope('predictor', reuse=reuse):

        conv_1 = conv2d_bn_relu('conv_1', input_, filters=16, kernel_size=3,
                                stride=1, padding='SAME')
        res_1 = residual(conv_1, output_dim=32, scope='res_pre_1')
        res_2 = residual(res_1, output_dim=32, scope='res_pre_2')
        res_3 = residual(res_2, output_dim=n_features, scope='res_pre_3')

        out = [None for _ in range(n_stacks)]

        inter = res_3

        for i in range(n_stacks):

            hg = hourglass(inter, output_dim=n_features, n=n_depth,
                           n_modules=n_modules, scope='hg_%d' % i)
            res = hg
            for j in range(n_modules):
                res = residual(res, n_features, scope='res_%d_%d' % (i, j))
            lin = conv2d_bn_relu('lin_%d' % i, res, filters=n_features,
                                 kernel_size=1, stride=1, padding='VALID',
                                 act_relu=False, bn=False)
            out[i] = conv2d_bn_relu('out_%d' % i, lin, filters=n_keypoints,
                                    kernel_size=1, stride=1, bn=False,
                                    act_relu=False)

            if i < n_stacks - 1:

                lin_ = conv2d_bn_relu('lin_out_%d' % i, lin,
                                      filters=n_features, kernel_size=1,
                                      stride=1, bn=False, act_relu=False)

                tmp_out = conv2d_bn_relu('tmp_out_%d' % i, out[i],
                                         filters=n_features, kernel_size=1,
                                         stride=1, bn=False, act_relu=False)
                inter = tf.add_n([inter + lin_ + tmp_out])

        output = tf.stack(out, axis=1)

    return output


def vgg_16(inputs, scope='vgg_16', reuse=False):

    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:

        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3],
                              scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection)

    return net, end_points
