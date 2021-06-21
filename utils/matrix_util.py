import tensorflow as tf


def from_tr(translation, rotation):
  # translation: ... x 3
  # rotation:    ... x 3 x 3
  T = tf.concat([rotation, translation[..., :, None]], axis=-1)
  br = tf.concat([tf.zeros_like(T[..., :1, :3]),
                  tf.ones_like(T[..., :1, :1])],
                 axis=-1)
  T = tf.concat([T, br], axis=-2)
  return T


def to_tr(pose):
  # translation: ... x 3
  # rotation:    ... x 3 x 3
  t = pose[..., :3, 3]
  r = pose[..., :3, :3]
  return t, r


def apply_transform(matrix, points):
  # matrix: ... x 4 x 4
  # points: ... x 3

  points = tf.concat([points, tf.ones_like(points[..., :1])], axis=-1)
  points = points[..., None]

  res = matrix @ points

  return res[..., :3, 0]