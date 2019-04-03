from src.archs import *

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of vgg-16 model')

hps = parser.parse_args()

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    with tf.Graph().as_default():

        x = tf.ones([16, 64, 64, 3], dtype=tf.float32)
        y = tf.ones([16, 12],  dtype=tf.float32)

        logits, endpoints_dict = vgg_16(x)

        variables_to_restore = slim.get_variables_to_restore(
            include=['vgg_16'])
        restorer = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restorer.restore(sess, hps.model_path)
