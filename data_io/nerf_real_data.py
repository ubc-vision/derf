import os

import numpy as np

import tensorflow as tf
from tensorflow_graphics.geometry import transformation

from utils import matrix_util
from data_io.read_model import read_cameras_binary, read_images_binary
from data_io.test_set_config import scene_to_test_img_files


def filter_ims_data(scene_name, ims_data, split):
  output = {}
  test_list = scene_to_test_img_files[scene_name]
  for key in ims_data:
    if split == 'test':
      if ims_data[key].name in test_list:
        output[key] = ims_data[key]
      else:
        pass
    elif split == 'train':
      if ims_data[key].name in test_list:
        pass
      else:
        output[key] = ims_data[key]
    else:
      raise ValueError
  if split == 'test':
    assert len(output) == len(test_list)
  elif split == 'train':
    assert len(output) == len(ims_data) - len(test_list)
  else:
    raise ValueError
  return output


def nerf_real_dataset(data_dir, split='train', scale=1.0):
  assert split in ['train', 'test'], f'No explicit split for {split}'
  cam = read_cameras_binary(os.path.join(data_dir, "sparse/0/cameras.bin"))[1]

  with tf.device("CPU:0"):
    if cam.model == "SIMPLE_RADIAL":
      f, ppx, ppy, _ = cam.params
      focal = tf.constant([[f, f]], dtype=tf.float32) * scale
      principal = tf.constant([[ppx, ppy]], dtype=tf.float32) * scale

    else:
      raise NotImplementedError(cam.model)

    poses_bounds = np.load(os.path.join(data_dir, "poses_bounds.npy"))
    poses_bounds = tf.cast(poses_bounds, tf.float32)
    mins, maxs = tf.transpose(poses_bounds[:, -2:])
    near = tf.reduce_min(mins)
    far = tf.reduce_max(maxs)

    ims_data = read_images_binary(os.path.join(data_dir, "sparse/0/images.bin"))
    ims_data = filter_ims_data('llff_' + os.path.basename(data_dir), ims_data,
                               split)

    rotations = []
    translations = []
    images = []

    for key in ims_data:
      data = ims_data[key]
      quat = np.array([data.qvec[1], data.qvec[2], data.qvec[3], data.qvec[0]])
      rot = transformation.rotation_matrix_3d.from_quaternion(quat)
      rotations.append(rot)
      translations.append(data.tvec)
      im = tf.image.decode_jpeg(tf.io.read_file(
          os.path.join(data_dir, "images", data.name)),
                                channels=3)
      resize_shape = (np.array(im.shape[:2]) * scale).astype(np.int)
      im = tf.image.resize(im, resize_shape, antialias=True)
      im = tf.cast(im, tf.float32) / 255.0
      im = tf.concat([im, tf.ones_like(im[..., :1])], axis=-1)
      images.append(im)

    translations = tf.stack(translations, axis=0)
    rotations = tf.stack(rotations, axis=0)

    translations = tf.cast(translations, tf.float32)
    rotations = tf.cast(rotations, tf.float32)

    world_2_cam = matrix_util.from_tr(translations, rotations)
    cam_2_world = tf.linalg.inv(world_2_cam)

    translations = cam_2_world[:, :3, 3]
    rotations = cam_2_world[:, :3, :3]

  ds = tf.data.Dataset.range(len(images))
  ims_ds = tf.data.Dataset.from_generator(lambda: images, tf.float32)

  def map_fn(i, rgba):
    pose = matrix_util.from_tr(translations[i], rotations[i])
    return pose, rgba

  ds = tf.data.Dataset.zip((ds, ims_ds)).map(map_fn)

  return ds, near, far, focal, principal
