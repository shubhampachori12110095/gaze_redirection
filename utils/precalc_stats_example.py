from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf

from utils import fid
from scipy.misc import imread
from skimage.transform import resize
from datetime import datetime
from tmp.config import config_celeba as config

########
# PATHS
########

# set path to training set images
data_path = '/cluster/scratch/zhehe/data/celebA/splits/train'

# path for storing the inception model
model_path = os.path.join(config['inception_model_path'],
                          'classify_image_graph_def.pb')

# path for where to store the statistics
output_path = config['inception_stat_path']

# number of samples to calculate fid score
num_samples = config['fid_batches'] * config['fid_batch_size']

# if you have downloaded and extracted
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
print("check for inception model..")
if os.path.exists(model_path):
    print("ok")
else:
    print("model doesn't exist! please download it first.")

# loads all images into memory (this might require a lot of RAM!)
print("load images..")
image_list = glob.glob(os.path.join(data_path, '*.jpg'))
print("images loaded.")
start_time = datetime.now()
random_idx = np.random.randint(0, len(image_list), num_samples)
images = np.array(
    [imread(
        str(image_list[idx])).astype(np.float32) for idx in random_idx])
end_time = datetime.now()
duration = end_time - start_time
print("%d images found and loaded in %.3f s" % (len(images),
                                                duration.total_seconds()))
images = [resize(_, [64, 64], preserve_range=True) for _ in images[:, 50:178, 25:153, :]]
print("create inception graph..")
# load the graph into the current TF graph
fid.create_inception_graph(model_path)
print("ok")

print("calculte FID stats..")
start_time = datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images,
                                                    sess,
                                                    batch_size=100,
                                                    verbose=True)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
end_time = datetime.now()
duration = end_time - start_time
print("finished in %.3f s" % duration.total_seconds())
