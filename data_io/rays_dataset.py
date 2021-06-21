import tensorflow as tf
from tensorflow_graphics.rendering.camera import perspective

from utils.matrix_util import *


def rays_dataset(images_ds, near, far, focal, principal, batch_size):

  def map_eyes(pose, rgb):
    del rgb
    return apply_transform(pose, tf.zeros((1, 3)))

  all_eyes = list(images_ds.map(map_eyes))

  n_eyes = len(all_eyes)

  eyes_min = tf.reduce_min(all_eyes, axis=0)
  eyes_max = tf.reduce_max(all_eyes, axis=0)
  eyes_bbox = tf.stack([eyes_min[0], eyes_max[0]], axis=0) / far


  def map_boundary_pts(pose, rgba):
    i = tf.convert_to_tensor([0, tf.shape(rgba)[0]])
    j = tf.convert_to_tensor([0, tf.shape(rgba)[1]])
    ij = tf.stack(tf.meshgrid(i, j, indexing="ij"), axis=-1)
    xy = tf.reshape(tf.cast(ij, tf.float32), (-1, 2))[:, ::-1]
    ip_points = perspective.ray(xy, focal, principal)
    ip_points = tf.concat([ip_points, tf.zeros((1, 3))], axis=0)
    ip_points = ip_points * far
    ws_ip_points = apply_transform(pose, ip_points)
    return ws_ip_points

  all_boundary_pts = list(images_ds.map(map_boundary_pts))
  all_boundary_pts = tf.concat(all_boundary_pts, axis=0)
  boundary_min = tf.reduce_min(all_boundary_pts, axis=0)
  boundary_max = tf.reduce_max(all_boundary_pts, axis=0)
  scene_bbox = tf.stack([boundary_min, boundary_max], axis=0) / far

  samples_per_im = 2**16

  def sample_rays(pose, rgba):
    n_rays = tf.shape(rgba)[0] * tf.shape(rgba)[1]
    i = tf.range(tf.shape(rgba)[0])
    j = tf.range(tf.shape(rgba)[1])
    ij = tf.stack(tf.meshgrid(i, j, indexing="ij"), axis=-1)
    xy = tf.reshape(tf.cast(ij, tf.float32), (-1, 2))[:, ::-1]
    ip_points = perspective.ray(xy, focal, principal)
    ws_ip_points = apply_transform(pose, ip_points) / far
    ws_eye = apply_transform(pose, tf.zeros((1, 3))) / far
    eyes_tiled = tf.tile(ws_eye, (n_rays, 1))
    ray_dirs, _ = tf.linalg.normalize(ws_ip_points - eyes_tiled, axis=-1)
    rays = tf.concat([eyes_tiled, ray_dirs], axis=-1)
    colors = tf.reshape(rgba, (-1, 4))

    inds = tf.random.uniform((samples_per_im,),
                             maxval=tf.cast(n_rays, tf.int64),
                             dtype=tf.int64)
    rays = tf.gather(rays, inds)
    rays = tf.concat([rays[:, :3], 1.5 * rays[:, 3:]], axis=-1)
    colors = tf.gather(colors, inds)

    return rays, colors

  ray_ds = images_ds.repeat().shuffle(n_eyes).map(sample_rays)

  def flatten_fn(*elem):
    return tf.data.Dataset.from_tensor_slices(elem)

  ray_ds = \
      ray_ds \
      .interleave(flatten_fn,
                  n_eyes,
                  num_parallel_calls=n_eyes,
                  deterministic=False) \
      .batch(batch_size) \
      .prefetch(32)

  return ray_ds, scene_bbox
