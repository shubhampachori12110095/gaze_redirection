"""training/testing model"""

from __future__ import division
import os
import glob
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.archs import discriminator, generator, vgg_16
from scipy.misc import imsave
from src.data_loader import ImageData, ImageAll
from tensorflow.contrib.data import shuffle_and_repeat, map_and_batch
from utils.ops import l1_loss, content_loss, style_loss, angular_error


class Model(object):
    """
    Model class.
    """
    def __init__(self, params):
        """init
        :param params: a Namespace() object."""

        self.params = params
        self.global_step = tf.Variable(0, dtype=tf.int32,
                                       trainable=False, name='global_step')
        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        (self.train_iter, self.valid_iter, self.test_iter,
         self.train_size) = self.data_loader()

        # building graph
        (self.x_r, self.angles_r, self.labels, self.x_t,
         self.angles_g) = self.train_iter.get_next()

        (self.x_valid_r, self.angles_valid_r, self.labels_valid,
         self.x_valid_t, self.angles_valid_g) = self.valid_iter.get_next()

        (self.x_test_r, self.angles_test_r, self.labels_test,
         self.x_test_t, self.angles_test_g) = self.test_iter.get_next()

        self.x_g = generator(self.x_r, self.angles_g)
        self.x_r_recon = generator(self.x_g, self.angles_g, reuse=True)

        self.angles_valid_g = tf.random_uniform(
            [params.batch_size, 2], minval=-1.0, maxval=1.0)

        self.x_valid_g = generator(self.x_valid_r, self.angles_valid_g,
                                   reuse=True)

        # reconstruction loss
        self.recon_loss = l1_loss(self.x_t, self.x_r_recon)

        # content loss and style loss
        self.c_loss, self.s_loss = self.feat_loss()

        # regression losses and adversarial losses
        (self.d_loss, self.g_loss, self.reg_d_loss, self.reg_g_loss,
         self.gp) = self.adv_loss()

        # update operations for generator and discriminator
        self.d_op, self.g_op = self.add_optimizer()

        # adding summaries
        self.summary = self.add_summary()

        # initialization operation
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

    def data_loader(self):
        """load traing and testing dataset"""

        hps = self.params

        image_data_class = ImageData(load_size=hps.image_size,
                                     channels=3,
                                     data_path=hps.data_path,
                                     ids=hps.ids)
        image_data_class.preprocess()

        train_dataset_num = len(image_data_class.train_images)
        test_dataset_num = len(image_data_class.test_images)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (image_data_class.train_images,
             image_data_class.train_angles_r,
             image_data_class.train_labels,
             image_data_class.train_images_t,
             image_data_class.train_angles_g))
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (image_data_class.test_images,
             image_data_class.test_angles_r,
             image_data_class.test_labels,
             image_data_class.test_images_t,
             image_data_class.test_angles_g))

        train_dataset = train_dataset.apply(
            shuffle_and_repeat(train_dataset_num)).apply(
            map_and_batch(image_data_class.image_processing,
                          hps.batch_size,
                          num_parallel_batches=8))

        valid_dataset = test_dataset.apply(
            shuffle_and_repeat(test_dataset_num)).apply(
            map_and_batch(image_data_class.image_processing,
                          hps.batch_size,
                          num_parallel_batches=8))

        test_dataset = test_dataset.apply(
            map_and_batch(image_data_class.image_processing,
                          hps.batch_size,
                          num_parallel_batches=8))

        train_dataset_iterator = train_dataset.make_one_shot_iterator()
        valid_dataset = valid_dataset.make_one_shot_iterator()
        test_dataset_iterator = test_dataset.make_one_shot_iterator()

        return (train_dataset_iterator,
                valid_dataset,
                test_dataset_iterator,
                train_dataset_num)

    def adv_loss(self):
        """
        build sub graph for discriminator and gaze estimator
        :return:
            d_loss: adversarial loss for training discriminator.
            g_loss: adcersarial loss ofr training generator.
            reg_loss_d: MSE loss for training gaze estimator
            reg_loss_g: MSE loss for training generator
            gp: gradient penalty
        """

        hps = self.params

        gan_real, reg_real = discriminator(hps, self.x_r)
        gan_fake, reg_fake = discriminator(hps, self.x_g, reuse=True)

        eps = tf.random_uniform(shape=[hps.batch_size, 1, 1, 1], minval=0.,
                                maxval=1.)
        interpolated = eps * self.x_r + (1. - eps) * self.x_g
        gan_inter, _ = discriminator(hps, interpolated, reuse=True)
        grad = tf.gradients(gan_inter, interpolated)[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean(tf.square(slopes - 1.))

        d_loss = -tf.reduce_mean(gan_real) + tf.reduce_mean(gan_fake) + 10. * gp
        g_loss = -tf.reduce_mean(gan_fake)

        reg_loss_d = tf.losses.mean_squared_error(self.angles_r, reg_real)
        reg_loss_g = tf.losses.mean_squared_error(self.angles_g, reg_fake)

        return d_loss, g_loss, reg_loss_d, reg_loss_g, gp

    def feat_loss(self):
        """
        build the sub graph of perceptual matching network
        return:
            c_loss: content loss
            s_loss: style loss
        """

        content_layers = ["vgg_16/conv5/conv5_3"]
        style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
                        "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

        _, endpoints_mixed = vgg_16(
            tf.concat([self.x_g, self.x_t], 0))

        c_loss = content_loss(endpoints_mixed, content_layers)
        s_loss = style_loss(endpoints_mixed, style_layers)

        return c_loss, s_loss

    def optimizer(self, lr):
        """
        return an optimizer
        :param lr: learning rate.
        :return: a tensorflow Optimizer object.
        """

        hps = self.params

        if hps.optimizer == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        if hps.optimizer == 'adam':
            return tf.train.AdamOptimizer(lr,
                                          beta1=hps.adam_beta1,
                                          beta2=hps.adam_beta2)
        raise AttributeError("attribute 'optimizer' is not assigned!")

    def add_optimizer(self):
        """
        add an optimizer.
        :return:
            g_op: update operation for generator.
            d_op: update operation for discriminator.
        """

        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        g_opt = self.optimizer(self.lr)
        d_opt = self.optimizer(self.lr)

        g_loss = (self.g_loss + 5.0 * self.reg_g_loss +
                  50.0 * self.recon_loss +
                  100.0 * self.s_loss + 100.0 * self.c_loss)
        d_loss = self.d_loss + 5.0 * self.reg_d_loss

        g_op = g_opt.minimize(loss=g_loss,
                              global_step=self.global_step,
                              var_list=g_vars)
        d_op = d_opt.minimize(loss=d_loss,
                              global_step=self.global_step,
                              var_list=d_vars)

        return d_op, g_op

    def add_summary(self):
        """
        add summary operation.
        :return:
            Tensor of dtype string.
        """

        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('reg_d_loss', self.reg_d_loss)
        tf.summary.scalar('reg_g_loss', self.reg_g_loss)
        tf.summary.scalar('gp', self.gp)
        tf.summary.scalar('lr', self.lr)
        tf.summary.scalar('c_loss', self.c_loss)
        tf.summary.scalar('s_loss', self.s_loss)

        tf.summary.image('real', (self.x_r + 1) / 2.0, max_outputs=5)
        tf.summary.image('fake', tf.clip_by_value(
            (self.x_g + 1) / 2.0, 0., 1.), max_outputs=5)
        tf.summary.image('recon', tf.clip_by_value(
            (self.x_r_recon + 1) / 2.0, 0., 1.), max_outputs=5)

        tf.summary.image('x_test', tf.clip_by_value(
            (self.x_valid_r + 1) / 2.0, 0., 1.), max_outputs=5)
        tf.summary.image('x_test_fake', tf.clip_by_value(
            (self.x_valid_g + 1) / 2.0, 0., 1.), max_outputs=5)

        summary_op = tf.summary.merge_all()

        return summary_op

    def train(self):
        """
        train the model and save checkpoints.
        """

        hps = self.params

        num_epoch = hps.epochs
        train_size = self.train_size
        batch_size = hps.batch_size
        learning_rate = hps.lr

        num_iter = train_size // batch_size

        summary_dir = os.path.join(hps.log_dir, 'summary')
        model_path = os.path.join(hps.log_dir, 'model.ckpt')

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:

            # init
            sess.run([self.init_op])

            summary_writer = tf.summary.FileWriter(summary_dir,
                                                   graph=sess.graph)

            saver = tf.train.Saver(max_to_keep=3)

            variables_to_restore = slim.get_variables_to_restore(
                include=['vgg_16'])
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, hps.vgg_path)

            try:

                for epoch in range(num_epoch):

                    print("Epoch: %d" % epoch)

                    if epoch >= hps.epochs / 2:

                        learning_rate = (2. - 2. * epoch / hps.epochs) * hps.lr

                    for it in range(num_iter):

                        feed_d = {self.lr: learning_rate}

                        sess.run([self.d_op], feed_dict=feed_d)

                        if it % 5 == 0:
                            sess.run(self.g_op, feed_dict=feed_d)

                        if it % hps.summary_steps == 0:

                            summary, global_step = sess.run(
                                [self.summary, self.global_step],
                                feed_dict=feed_d)
                            summary_writer.add_summary(summary, global_step)
                            summary_writer.flush()
                            saver.save(sess, model_path,
                                       global_step=global_step)

            except KeyboardInterrupt:
                print("stop training")

    def eval(self):
        """
        generate eye patch images given certain eye gaze movement trajectary.
        """

        hps = self.params

        checkpoint = tf.train.latest_checkpoint(hps.log_dir)

        x_test_r, _, _, _, _ = self.valid_iter.get_next()

        images = tf.placeholder(tf.float32, shape=[None, hps.image_size,
                                                   hps.image_size, 3],
                                name='image')
        angles = tf.placeholder(tf.float32, shape=[None, 2], name='angles')

        x_test_g = generator(images, angles, reuse=True)

        saver = tf.train.Saver()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as test_sess:

            with test_sess.graph.as_default():

                saver.restore(test_sess, checkpoint)
                real_dir = os.path.join(hps.imgs_dir, 'real')
                gene_dir = os.path.join(hps.imgs_dir, 'genes')
                os.makedirs(real_dir)
                os.makedirs(gene_dir)

                real_imgs = test_sess.run(x_test_r)

                theta = np.linspace(0, 2 * np.pi, 120)
                x_cord = np.sin(theta)
                y_cord = np.cos(theta)

                x_cord = np.concatenate((np.linspace(-1, 1, 40),
                                         np.linspace(1, -1, 40),
                                         np.linspace(-1, 1, 40)))

                y_cord = np.concatenate((np.repeat(1, 40),
                                         np.linspace(1, -1, 40),
                                         np.repeat(-1, 40)))

                fake_labels = np.column_stack((x_cord, y_cord))

                for idx in range(hps.batch_size):

                    imsave(os.path.join(real_dir, '%d.jpg' % idx),
                           real_imgs[idx])

                    imgs_batch = np.tile(real_imgs[idx],
                                         [120, 1, 1, 1])

                    fake_imgs = test_sess.run(x_test_g,
                                              feed_dict={images: imgs_batch,
                                                         angles: fake_labels})
                    imgs_batch = (imgs_batch + 1.0) / 2.0
                    fake_imgs = np.clip((fake_imgs + 1.0) / 2.0, 0.0, 1.0)
                    for i in range(120):
                        imsave(os.path.join(gene_dir, '%d_%d.jpg' % (idx, i)),
                               fake_imgs[i])

    def load_test_data(self):

        hps = self.params

        images = []
        sides = []
        image_files = glob.glob(os.path.join(hps.test_dir, '*.jpg'))
        for f in image_files:

            side = f.split('/')[-1].split('.')[0].split('_')[-1]

            images.append(f)
            sides.append(side)

        return images, sides

    def test_data_loader(self):

        hps = self.params

        def image_processing(filename, side):

            def _to_image(file_name):
                x = tf.read_file(file_name)
                img = tf.image.decode_jpeg(x, channels=3)
                img = tf.image.resize_images(img,
                                             [hps.image_size, hps.image_size])
                img = tf.cast(img, tf.float32) / 127.5 - 1.0

                return img

            img = _to_image(filename)

            return img, side

        test_images, test_sides = self.load_test_data()
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images,
                                                           test_sides))
        test_dataset = test_dataset.apply(
                map_and_batch(image_processing,
                              hps.batch_size,
                              num_parallel_batches=8))
        test_dataset_iterator = test_dataset.make_one_shot_iterator()

        return test_dataset_iterator

    def test(self):

        hps = self.params

        pitch_angle = np.array([-10, 0, 10]) / 10.0
        yaw_angle = np.array([-15, -10, -5, 0, 5, 10, 15]) / 15.0

        batch_size = pitch_angle.shape[0] * yaw_angle.shape[0]

        checkpoint = tf.train.latest_checkpoint(hps.log_dir)
        test_iter = self.test_data_loader()
        x_test, sides_test = test_iter.get_next()

        images = tf.placeholder(tf.float32,
                                shape=[batch_size, hps.image_size,
                                       hps.image_size, 3],
                                name='image')
        angles = tf.placeholder(tf.float32,
                                shape=[batch_size, 2],
                                name='angles')

        x_fake = generator(images, angles, reuse=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        with tf.Session(config=tf_config) as test_sess:

            with test_sess.graph.as_default():
                saver.restore(test_sess, checkpoint)

                imgs_dir = os.path.join(hps.log_dir, 'wild_images')
                if not os.path.exists(imgs_dir):
                    os.mkdir(imgs_dir)

                real_dir = os.path.join(imgs_dir, 'real')
                gene_dir = os.path.join(imgs_dir, 'genes')

                os.makedirs(real_dir)
                os.makedirs(gene_dir)

                real_imgs, real_sides = test_sess.run([x_test, sides_test])

                theta = np.zeros([batch_size, 2], dtype=np.float32)

                for i, yaw in enumerate(yaw_angle):
                    for j, pitch in enumerate(pitch_angle):
                        theta[pitch_angle.shape[0] * i + j, 0] = yaw
                        theta[pitch_angle.shape[0] * i + j, 1] = pitch

                for idx in range(hps.batch_size):

                    imgs_batch = np.tile(real_imgs[idx],
                                         [batch_size, 1, 1, 1])

                    fake_imgs = test_sess.run(x_fake,
                                              feed_dict={images: imgs_batch,
                                                         angles: theta})

                    imgs_batch = (imgs_batch + 1.0) / 2.0
                    fake_imgs = np.clip((fake_imgs + 1.0) / 2.0, 0.0, 1.0)

                    for i in range(21):

                        imsave(os.path.join(real_dir, '%d_%d_%s.jpg' % (idx, i, real_sides[idx])),
                               imgs_batch[i])
                        imsave(os.path.join(gene_dir, '%d_%d_%s.jpg' % (idx, i, real_sides[idx])),
                               fake_imgs[i])

    def quanti_eval(self):
        """
        generate random samples.
        """
        hps = self.params

        checkpoint = tf.train.latest_checkpoint(hps.log_dir)

        x_fake = generator(self.x_test_r, self.angles_test_g, reuse=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        with tf.Session(config=tf_config) as test_sess:

            with test_sess.graph.as_default():
                saver.restore(test_sess, checkpoint)

                imgs_dir = os.path.join(hps.log_dir, 'quanti_eval')
                if not os.path.exists(imgs_dir):
                    os.mkdir(imgs_dir)

                tar_dir = os.path.join(imgs_dir, 'targets')
                gene_dir = os.path.join(imgs_dir, 'genes')
                real_dir = os.path.join(imgs_dir, 'reals')

                os.makedirs(tar_dir)
                os.makedirs(gene_dir)
                os.makedirs(real_dir)

                try:
                    i = 0
                    while True:
                        real_imgs, target_imgs, fake_imgs, a_r, a_t = test_sess.run(
                            [self.x_test_r, self.x_test_t, x_fake,
                             self.angles_test_r, self.angles_test_g])
                        a_t = a_t * np.array([15, 10])
                        a_r = a_r * np.array([15, 10])
                        delta = angular_error(a_t, a_r)

                        for j in range(hps.batch_size):
                            imsave(os.path.join(
                                tar_dir, '%d_%d_%.3f_H%d_V%d.jpg' % (i, j, delta[j], a_t[j][0], a_t[j][1])),
                                   target_imgs[j])
                            imsave(os.path.join(
                                gene_dir, '%d_%d_%.3f_H%d_V%d.jpg' % (i, j, delta[j], a_t[j][0], a_t[j][1])),
                                   fake_imgs[j])
                            imsave(os.path.join(
                                real_dir, '%d_%d_%.3f_H%d_V%d.jpg' % (i, j, delta[j], a_t[j][0], a_t[j][1])),
                                   real_imgs[j])

                        i = i + 1
                except tf.errors.OutOfRangeError:
                    logging.info("quanti_eval finished.")

    def dataset_gene(self):

        hps = self.params

        image_data_class = ImageAll(load_size=hps.image_size,
                                    channels=3,
                                    data_path=hps.data_path)
        image_data_class.preprocess()

        dataset = tf.data.Dataset.from_tensor_slices(
            (image_data_class.images,
             image_data_class.angles,
             image_data_class.ids,
             image_data_class.suffix))

        dataset = dataset.apply(
            map_and_batch(image_data_class.image_processing,
                          hps.batch_size,
                          num_parallel_batches=8))

        dataset_iterator = dataset.make_one_shot_iterator()

        image, angle, index, suffix = dataset_iterator.get_next()

        checkpoint = tf.train.latest_checkpoint(hps.log_dir)

        image_syn = generator(image, angle, reuse=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        with tf.Session(config=tf_config) as test_sess:

            with test_sess.graph.as_default():
                saver.restore(test_sess, checkpoint)

                imgs_dir = os.path.join(hps.log_dir, 'dataset')
                if not os.path.exists(imgs_dir):
                    os.mkdir(imgs_dir)

                try:
                    while True:
                        imgs, gaze, idx, suffix_str = test_sess.run(
                            [image_syn, angle, index, suffix])
                        gaze = gaze * np.array([15, 10])
                        for j in range(hps.batch_size):
                            imsave(os.path.join(
                                imgs_dir, '%s_%.3fV_%.3fH_%s.jpg' % (idx[j], gaze[j][1], gaze[j][0], suffix_str[j])), imgs[j])

                except tf.errors.OutOfRangeError:
                    logging.info("dataset_gene finished.")
