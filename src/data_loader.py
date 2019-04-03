import os
import tensorflow as tf

from tensorflow.contrib.data import shuffle_and_repeat, map_and_batch


class ImageData(object):

    def __init__(self, load_size, channels, data_path, ids):
        self.load_size = load_size
        self.channels = channels
        self.ids = ids

        self.data_path = data_path
        file_names = [f for f in os.listdir(data_path)
                      if f.endswith('.jpg')]
        self.file_dict = dict()
        for f_name in file_names:
            key = f_name.split('.')[0].split('_')[0]
            side = f_name.split('.')[0].split('_')[-1]
            key = key + '_' + side
            if key not in self.file_dict.keys():
                self.file_dict[key] = []
                self.file_dict[key].append(f_name)
            else:
                self.file_dict[key].append(f_name)

        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.test_images_t = []
        self.test_angles_g = []

    def image_processing(self, filename, angles_r, labels, filename_t,
                         angles_g):

        def _to_image(file_name):

            x = tf.read_file(file_name)
            img = tf.image.decode_jpeg(x, channels=self.channels)
            img = tf.image.resize_images(img, [self.load_size, self.load_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0

            return img

        image = _to_image(filename)
        image_t = _to_image(filename_t)

        return image, angles_r, labels, image_t, angles_g

    def preprocess(self):

        for key in self.file_dict.keys():

            idx = int(key.split('_')[0])
            flip = 1
            if key.split('_')[-1] == 'R':
                flip = -1

            for f_r in self.file_dict[key]:

                file_path = os.path.join(self.data_path, f_r)

                h_angle_r = flip * float(
                    f_r.split('_')[-2].split('H')[0]) / 15.0
                v_angle_r = float(
                    f_r.split('_')[-3].split('V')[0]) / 10.0

                for f_g in self.file_dict[key]:

                    file_path_t = os.path.join(self.data_path, f_g)

                    h_angle_g = flip * float(
                        f_g.split('_')[-2].split('H')[0]) / 15.0
                    v_angle_g = float(
                        f_g.split('_')[-3].split('V')[0]) / 10.0

                    if idx <= self.ids:
                        self.train_images.append(file_path)
                        self.train_angles_r.append([h_angle_r, v_angle_r])
                        self.train_labels.append(idx - 1)
                        self.train_images_t.append(file_path_t)
                        self.train_angles_g.append([h_angle_g, v_angle_g])
                    else:
                        self.test_images.append(file_path)
                        self.test_angles_r.append([h_angle_r, v_angle_r])
                        self.test_labels.append(idx - 1)
                        self.test_images_t.append(file_path_t)
                        self.test_angles_g.append([h_angle_g, v_angle_g])

        print('\nFinished preprocessing the dataset...')


class ImageAll(object):

    def __init__(self, load_size, channels, data_path):
        self.load_size = load_size
        self.channels = channels
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]

        self.images = []
        self.angles = []
        self.ids = []
        self.suffix = []

    def image_processing(self, filename, angle, index, suffix):

        def _to_image(file_name):

            x = tf.read_file(file_name)
            img = tf.image.decode_jpeg(x, channels=self.channels)
            img = tf.image.resize_images(img, [self.load_size, self.load_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0

            return img

        image = _to_image(filename)

        return image, angle, index, suffix

    def preprocess(self):

        i = 0

        for file in self.files:
            path = os.path.join(self.data_path, file)
            fields = file.replace('.jpg', '').split('_')
            index = fields[0] + '_' + fields[-1]

            if int(fields[0]) > 50:
                i = i + 1
                for h_angle in [-15, -10, -5, 0, 5, 10, 15]:
                    for v_angle in [-10, 0, 10]:
                        self.images.append(path)
                        self.angles.append([h_angle / 15.0, v_angle / 10.0])
                        self.ids.append(index)
                        self.suffix.append(str(i))
        print("preprocessing finished!")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path of training data')

    params = parser.parse_args()

    dummy_data = ImageAll(64, 3, params.data_path)
    dummy_data.preprocess()
    print(dummy_data.images.__len__())
    print(dummy_data.images[0])
    print(dummy_data.angles[0])
    print(dummy_data.ids[0])

    train_dataset_num = len(dummy_data.images)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dummy_data.images,
         dummy_data.angles,
         dummy_data.ids))

    train_dataset = train_dataset.apply(
        map_and_batch(dummy_data.image_processing,
                      16,
                      num_parallel_batches=8))

    train_dataset_iterator = train_dataset.make_one_shot_iterator()

    (image, angle, id) = train_dataset_iterator.get_next()

    print(image.get_shape().as_list())
    print(angle.get_shape().as_list())
    print(id.get_shape().as_list())

    # import csv
    #
    # with open('./record.csv', 'w') as f:
    #     for line in zip(dummy_data.train_images,
    #                     dummy_data.train_angles_r,
    #                     dummy_data.train_labels,
    #                     dummy_data.train_images_t,
    #                     dummy_data.train_angles_g):
    #         writer = csv.writer(f, delimiter=',')
    #         writer.writerow(line)
