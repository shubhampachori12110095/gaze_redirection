"""Main script."""

import argparse
import os
import logging.config
import shutil
import yaml

from src.model import Model

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_logging(default_path='config/logging.yaml',
                  default_level=logging.INFO):
    """
    Setup logging configuration
    """
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        print('the input path doesn\'t exist')


setup_logging()

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train',
                    help='running mode')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--data_path', type=str, help='path of faces')

# optimizer params
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd', 'adagrad', 'rmsprop'])
parser.add_argument('--adam_beta1', type=float, default=0.5,
                    help='value of adam beta 1')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='value of adam beta 2')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')

# training params
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--summary_steps', type=int, default=500,
                    help='summary steps')


# dataset params
parser.add_argument('--image_size', type=int, default=64,
                    help='size of cropped images')
parser.add_argument('--ids', type=int, default=50,
                    help='number of identities for training')

# evaluation dir
parser.add_argument('--log_dir', type=str, help='path of eval checkpoint')
parser.add_argument('--vgg_path', type=str, help='path of vgg model')

# test dir
parser.add_argument('--test_dir', type=str, help='path of test images')

params = parser.parse_args()


def create_dir(path):

    if not os.path.exists(path):
        os.mkdir(path)


if params.mode == 'train':

    create_dir(params.log_dir)

    code_path = os.path.join(params.log_dir, 'code')
    if os.path.exists(code_path):
        os.removedirs(code_path)

    shutil.copytree('.', code_path, ignore=shutil.ignore_patterns('*.pyc',
                                                                  'lsf.*'))
elif params.mode == 'eval':
    assert params.log_dir, "run in evaluation mode, attribute log_dir must" \
                            " be specified"
    params.imgs_dir = os.path.join(params.log_dir, 'images')
    if os.path.exists(params.imgs_dir):
        shutil.rmtree(params.imgs_dir)
        os.mkdir(params.imgs_dir)

model = Model(params)

if params.mode == 'train':
    model.train()
elif params.mode == 'eval':
    model.eval()
elif params.mode == 'debug':
    pass
elif params.mode == 'test':
    model.test()
elif params.mode == 'quanti_eval':
    model.quanti_eval()
elif params.mode == 'dataset_gene':
    model.dataset_gene()

