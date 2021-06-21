import tensorflow as tf


def img2mse(x, y):
  return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x):
  return -10. * tf.math.log(x) / tf.math.log(10.)


def weights_to_rgb(weights):
  leading_dims = tf.rank(weights) - 1
  heads = tf.shape(weights)[-1]
  hue = tf.linspace(0.0, 1.0, heads + 1)[:-1]
  hue = tf.reshape(
      hue, tf.concat([tf.ones((leading_dims), dtype=tf.int32), [-1]], axis=0))
  hue *= tf.ones_like(weights)
  saturation = tf.ones_like(weights)
  value = weights
  weights_hsv = tf.stack([hue, saturation, value], axis=-1)
  weights_rgb = tf.image.hsv_to_rgb(weights_hsv)
  weights_rgb = tf.reduce_sum(weights_rgb, axis=-2)
  return weights_rgb


def imwrite(fname, im):
  data = tf.image.encode_png(
      tf.cast(255.0 * tf.clip_by_value(im, 0.0, 1.0), tf.uint8))
  tf.io.write_file(fname, data)