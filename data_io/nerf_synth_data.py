import os
import json

import numpy as np
import imageio

import tensorflow as tf

from utils.matrix_util import *


def nerf_synth_dataset(data_dir, split='train', scale=1.0):
  assert split in ['train', 'val', 'test']

  with open(os.path.join(data_dir, f'transforms_{split}.json'), 'r') as fp:
    model_meta = json.load(fp)

  test_image_path = os.path.join(data_dir,
                                 model_meta['frames'][0]['file_path'] + '.png')
  height, width = imageio.imread(test_image_path).shape[:2]
  fov = float(model_meta['camera_angle_x'])
  f = .5 * width / np.tan(.5 * fov)
  focal = tf.constant([[f, f]], dtype=tf.float32) * scale
  principal = tf.constant([[width / 2, height / 2]], dtype=tf.float32) * scale

  images = []
  poses = []
  for frame_meta in model_meta['frames']:
    cur_image_path = os.path.join(data_dir, frame_meta['file_path'] + '.png')
    im = tf.image.decode_png(tf.io.read_file(cur_image_path), channels=4)
    resize_shape = (np.array(im.shape[:2]) * scale).astype(np.int)
    im = tf.image.resize(im, resize_shape)
    im = tf.cast(im, tf.float32) / 255.0
    images.append(im)
    transf = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1.],
    ])
    poses.append(
        (np.array(frame_meta['transform_matrix']) @ transf).astype(np.float32))

  poses = tf.stack(poses, axis=0)

  ds = tf.data.Dataset.range(len(images))
  ims_ds = tf.data.Dataset.from_generator(lambda: images, tf.float32)

  def map_fn(i, rgba):
    pose = poses[i]
    return pose, rgba

  ds = tf.data.Dataset.zip((ds, ims_ds)).map(map_fn)

  far = 6.0
  near = 2.0

  return ds, near, far, focal, principal
