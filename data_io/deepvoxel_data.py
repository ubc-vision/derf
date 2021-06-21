import os

import tensorflow as tf

from utils.matrix_util import *


def deepvoxel_dataset(dataset_dir, split='train', scale=1.0):
  assert split in ['train', 'val', 'test']
  assert 'train' not in dataset_dir
  assert 'validation' not in dataset_dir
  assert 'test' not in dataset_dir
  if split == 'val':
    split == 'validation'

  dataset_dir = os.path.join(os.path.dirname(dataset_dir), split,
                             os.path.basename(dataset_dir))
  assert os.path.isdir(dataset_dir)

  with open(os.path.join(dataset_dir, "intrinsics.txt")) as ifd:
    vals = list(list(float(v) for v in l.split()) for l in ifd.readlines())
    focal = tf.constant([(vals[0][0] / vals[4][0]) * 512] * 2) * scale
    principal = tf.constant([255.5, 255.5]) * scale

    near = vals[2][0]

  pose_dir = os.path.join(dataset_dir, "pose")
  assert os.path.isdir(pose_dir)
  pose_path = sorted(os.path.join(pose_dir, f) for f in os.listdir(pose_dir))

  depth_dir = os.path.join(dataset_dir, "depth")
  has_depth = os.path.exists(depth_dir)
  if has_depth:
    depth_path = sorted(
        os.path.join(depth_dir, f) for f in os.listdir(depth_dir))
  else:
    depth_path = list("" for _ in pose_path)

  rgb_dir = os.path.join(dataset_dir, "rgb")
  assert os.path.isdir(rgb_dir)
  rgb_path = sorted(os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir))

  ds = tf.data.Dataset.from_tensor_slices((pose_path, depth_path, rgb_path))

  def load(pose_path, depth_path, rgb_path):
    pose = tf.strings.to_number(tf.strings.split(tf.io.read_file(pose_path)))
    pose = tf.reshape(pose, (4, 4))

    rgba = tf.io.decode_png(tf.io.read_file(rgb_path), 4)
    resize_shape = tf.cast(
        tf.cast(tf.shape(rgba)[:2], tf.float32) * scale, tf.int32)
    rgba = tf.image.resize(rgba, resize_shape)
    rgba = tf.cast(rgba, tf.float32) / 255.0

    if has_depth:
      depth = tf.io.decode_png(tf.io.read_file(depth_path), 1, dtype=tf.uint16)
      d_alpha = 1.0 - tf.cast(tf.equal(depth, tf.reduce_min(depth)), tf.float32)
      rgba = tf.concat([rgba[..., :3], d_alpha], axis=-1)

    return pose, rgba

  ds = ds.map(load)

  def map_eyes(pose, rgba):
    del rgba
    return apply_transform(pose, tf.zeros((1, 3)))

  all_eyes = list(ds.map(map_eyes))
  eyes_min = tf.reduce_min(all_eyes, axis=0)
  eyes_max = tf.reduce_max(all_eyes, axis=0)
  far = tf.norm(eyes_max - eyes_min).numpy()

  return ds, near, far, focal, principal
