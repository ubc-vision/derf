import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_graphics.rendering.camera import perspective

from utils.matrix_util import *


class PositionalEncoding(tf.Module):

  def __init__(self, depth=12):
    self.depth = depth

  def __call__(self, inputs):
    return tf.concat([tf.sin(2.0**i * inputs) for i in range(self.depth)] +
                     [tf.cos(2.0**i * inputs) for i in range(self.depth)],
                     axis=-1)


class RadianceFieldBase(tf.Module):

  def gen_ray_samples(self, rays, samples, pdf=None, deterministic=False):
    N = tf.shape(rays)[0]
    origins = rays[:, :3]
    target_offset = rays[:, 3:]

    if pdf is not None:
      imp_samples = tf.shape(pdf)[-2]

    else:
      imp_samples = 0

    fine_samples = samples - imp_samples

    t = tf.linspace(0.0, 1.0, fine_samples)[None]
    if not deterministic:
      t += tf.random.uniform((N, fine_samples)) / float(fine_samples)

    else:
      t += tf.zeros((N, fine_samples))

    if pdf is not None:
      pdfx = tf.concat([tf.zeros_like(pdf[..., :1, 0]), pdf[..., 0]], axis=-1)
      pdfy = pdf[..., 1] + 1e-5
      pdfy /= tf.reduce_sum(pdfy, axis=-1, keepdims=True)
      cdf = tf.cumsum(pdfy, axis=-1)
      cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)

      u = tf.linspace(0.0, 1.0, imp_samples)
      if deterministic:
        u = u[None] + tf.zeros((N, imp_samples))
      else:
        u = u[None] + tf.random.uniform((N, imp_samples)) / float(imp_samples)

      u = tf.clip_by_value(u, 0.0, 0.9999)

      inds = tf.searchsorted(cdf, u, side='right')
      below = tf.maximum(0, inds - 1)
      above = tf.minimum(tf.shape(cdf)[-1] - 1, inds)
      inds_g = tf.stack([below, above], -1)
      cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)
      pdfx_g = tf.gather(pdfx,
                         inds_g,
                         axis=-1,
                         batch_dims=len(inds_g.shape) - 2)

      denom = (cdf_g[..., 1] - cdf_g[..., 0])
      denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
      a = (u - cdf_g[..., 0]) / denom
      a = tf.clip_by_value(a, 0.0, 1.0)
      ti = pdfx_g[..., 0] + a * (pdfx_g[..., 1] - pdfx_g[..., 0])

      t = tf.concat([t, ti], axis=-1)
      t = tf.sort(t, axis=-1)

    dt = t[:, 1:] - t[:, :-1]
    dt = tf.concat([dt, dt[:, -1:]], axis=-1)
    xyz = origins[:, None, :] + t[:, :, None] * target_offset[:, None, :]

    dir_in, lengths = tf.linalg.normalize(target_offset, axis=-1)
    dir_in = tf.tile(dir_in[:, None, :], (1, samples, 1))

    return xyz, dir_in, lengths, dt, t

  def trace_rays(self, rays, samples, pdf=None, deterministic=False, **kwargs):
    xyz, dir_in, lengths, dt, t = self.gen_ray_samples(
        rays, samples, pdf=pdf, deterministic=deterministic)

    if deterministic:
      density_noise = 0.0

    else:
      density_noise = 1e0

    radiance_samples, density_samples, attention_samples = self.sample_field(
        xyz, dir_in, density_noise, **kwargs)

    delta = lengths * dt

    quantity = delta * density_samples

    extinction = tf.exp(tf.cumsum(-quantity, axis=-1, exclusive=True))[...,
                                                                       None]
    cross_section = 1.0 - tf.exp(-quantity)[..., None]

    radiance = tf.reduce_sum(extinction * cross_section * radiance_samples,
                             axis=-2)
    alpha = 1.0 - tf.exp(tf.reduce_sum(-quantity, axis=-1, keepdims=True))
    rgba = tf.concat([radiance, alpha], axis=-1)

    depth_samples = t[..., None] * lengths[..., None]
    norm = tf.reduce_sum(extinction * cross_section, axis=-2) + 1e-7
    depth = tf.reduce_sum(extinction * cross_section * depth_samples,
                          axis=-2) / norm
    attention = tf.reduce_sum(extinction * cross_section * attention_samples,
                              axis=-2) / norm

    return rgba, depth, attention

  def render(self,
             pose,
             farplane,
             focal,
             principal,
             res,
             samples,
             chunk=64,
             **kwargs):
    i = tf.range(res[0])
    j = tf.range(res[1])
    ij = tf.reshape(tf.stack(tf.meshgrid(i, j, indexing="ij"), axis=-1),
                    (-1, 2))
    xy = tf.reshape(tf.cast(ij, tf.float32), (-1, 2))[:, ::-1]
    ip_points = perspective.ray(xy, focal, principal)
    ws_ip_points = apply_transform(pose, ip_points)
    ws_eye = apply_transform(pose, tf.zeros((1, 3)))
    eyes_tiled = tf.tile(ws_eye, (res[0] * res[1], 1))
    ray_dirs, _ = tf.linalg.normalize(ws_ip_points - eyes_tiled, axis=-1)
    rays = tf.concat([eyes_tiled, farplane * ray_dirs], axis=-1)

    chunk = 64

    def cond(i, rgba, depth, attention):
      return i < (1 + (res[0] * res[1]) // chunk)

    def body(i, rgba, depth, attention):
      rgba_i, depth_i, attention_i = self.trace_rays(rays[i * chunk:(i + 1) *
                                                          chunk],
                                                     samples=samples,
                                                     deterministic=True,
                                                     **kwargs)

      rgba = tf.concat([rgba, rgba_i], axis=0)
      depth = tf.concat([depth, depth_i], axis=0)
      attention = tf.concat([attention, attention_i], axis=0)

      return [i + 1, rgba, depth, attention]

    lvars = [0, tf.zeros((0, 4)), tf.zeros((0, 1)), tf.zeros((0, self.n_heads))]
    _, rgba, depth, attention = tf.while_loop(cond,
                                              body,
                                              lvars,
                                              parallel_iterations=1,
                                              shape_invariants=[
                                                  tf.TensorShape([]),
                                                  tf.TensorShape([None, 4]),
                                                  tf.TensorShape([None, 1]),
                                                  tf.TensorShape(
                                                      [None, self.n_heads])
                                              ])

    rgba = tf.reshape(rgba, (res[0], res[1], 4))
    depth = tf.reshape(depth, (res[0], res[1], 1))
    attention = tf.reshape(attention, (res[0], res[1], self.n_heads))

    return rgba, depth, attention


class RadianceField(RadianceFieldBase):

  def __init__(self,
               units=256,
               pos_depth=8,
               rad_depth=1,
               pos_feature=PositionalEncoding(),
               dir_feature=PositionalEncoding(3),
               disable_view_dependence=False):
    self.pos_feature = pos_feature
    self.dir_feature = dir_feature
    self.disable_view_dependence = disable_view_dependence
    self.n_heads = 1

    self.density_layers_first = []
    for _ in range(min(pos_depth, 5)):
      self.density_layers_first += [
          layers.Dense(units),
          layers.ReLU(),
      ]

    self.density_layers_second = []
    for _ in range(pos_depth - 5):
      self.density_layers_second += [
          layers.Dense(units),
          layers.ReLU(),
      ]

    self.radiance_layers = []

    for _ in range(rad_depth):
      self.radiance_layers += [
          layers.Dense(units),
          layers.ReLU(),
      ]

    self.to_density = layers.Dense(1)
    self.to_radiance = layers.Dense(3)

  def sample_field(self,
                   positions,
                   directions,
                   density_noise=1e0,
                   mask_fn=None):
    coord_in = self.pos_feature(positions)
    dir_in = self.dir_feature(directions)

    net = coord_in

    for l in self.density_layers_first:
      net = l(net)

    net = tf.concat([net, tf.cast(coord_in, net.dtype)], axis=-1)

    for l in self.density_layers_second:
      net = l(net)

    density = self.to_density(net)[..., 0]
    density = tf.cast(density, tf.float32)
    density += tf.random.normal(tf.shape(density)) * density_noise
    density = tf.nn.relu(density)

    if mask_fn is not None:
      density *= tf.cast(mask_fn(positions), tf.float32)

    if not self.disable_view_dependence:
      net = tf.concat([net, tf.cast(dir_in, net.dtype)], axis=-1)

    for l in self.radiance_layers:
      net = l(net)

    radiance = self.to_radiance(net)
    radiance = tf.cast(radiance, tf.float32)
    radiance = tf.nn.sigmoid(radiance)

    return radiance, density, tf.ones_like(radiance[..., :1])
