# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
"""
This is the resnet structure
"""
from __future__ import division

import tensorflow as tf


BN_EPSILON = 0.001
W_DECAY = 0.0002


def print_tensor(x):

    print(x.name, x.get_shape().as_list())


def activation_summary(x):

    """
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    """

    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape,
                     initializer=tf.contrib.layers.xavier_initializer()):

    """
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    layers.
    :return: The created variable
    """

    regularizer = tf.contrib.layers.l2_regularizer(scale=W_DECAY)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    """
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    """
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(
        name='fc_weights', shape=[input_dim, num_labels],
        initializer=tf.variance_scaling_initializer(scale=1.0,
                                                    distribution='uniform'))
    fc_b = create_variables(name='fc_bias', shape=[num_labels],
                            initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable(
        'beta', dimension, tf.float32,
        initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(
        'gamma', dimension, tf.float32,
        initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def cls_loss(logits, labels):

    loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                    logits=logits)
        )
    )

    return loss


def compute_accuracy(logits, labels):

    x = tf.nn.sigmoid(logits)
    predicted = tf.cast(tf.greater(x, 0.5), tf.float32)
    correct = tf.cast(tf.equal(predicted, labels), tf.float32)
    accuracy = tf.reduce_mean(correct) * 100.0
    accuracy_list = tf.reduce_mean(correct, axis=0) * 100.0

    return accuracy, accuracy_list


def inference(input_tensor_batch, output_dim, n, reuse=False):
    """
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n + 2n + 1 = 8n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    """

    with tf.variable_scope('res-net', reuse=reuse):

        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 64], 1)
            activation_summary(conv0)
            layers.append(conv0)

        for i in range(n):
            with tf.variable_scope('conv1_%d' %i, reuse=reuse):
                if i == 0:
                    conv1 = residual_block(layers[-1], 64, first_block=True)
                else:
                    conv1 = residual_block(layers[-1], 64)
                activation_summary(conv1)
                layers.append(conv1)

        for i in range(n):
            with tf.variable_scope('conv2_%d' %i, reuse=reuse):
                conv2 = residual_block(layers[-1], 128)
                activation_summary(conv2)
                layers.append(conv2)

        for i in range(n):
            with tf.variable_scope('conv3_%d' %i, reuse=reuse):
                conv3 = residual_block(layers[-1], 256)
                activation_summary(conv3)
                layers.append(conv3)

        for i in range(n):
            with tf.variable_scope('conv4_%d' % i, reuse=reuse):
                conv4 = residual_block(layers[-1], 512)
                activation_summary(conv4)
                layers.append(conv4)

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])
            output = output_layer(global_pool, output_dim)
            layers.append(output)

        for _ in layers:
            print_tensor(_)

    return layers[-1]
