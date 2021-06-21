import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_graphics.rendering.camera import perspective

from scipy.spatial import Delaunay

from utils.matrix_util import *

from model.nerf import RadianceField, RadianceFieldBase, PositionalEncoding


class MLPDecomposition(tf.Module):

  def __init__(self,
               heads,
               units=64,
               depth=3,
               pos_feature=PositionalEncoding(5)):
    self.n_heads = heads
    self.pos_feature = pos_feature

    self.layers = []
    for _ in range(depth):
      self.layers += [
          layers.Dense(units),
          layers.ReLU(),
      ]
    self.layers += [layers.Dense(self.n_heads)]

  def __call__(self, positions):
    net = self.pos_feature(positions)

    for l in self.layers:
      net = l(net)

    weights = tf.nn.softmax(tf.cast(net, tf.float32))

    return weights


class VoronoiDecomposition(tf.Module):

  def __init__(self, scene_bbox, n_heads):
    self.n_heads = n_heads

    self.temperature = tf.Variable(1.0, trainable=False)

    self.center_scale = 1.0
    init_centers = self.center_scale * tf.random.uniform(
        (n_heads, 3)) * (scene_bbox[1] - scene_bbox[0]) + scene_bbox[0]
    self.head_centers = tf.Variable(init_centers, trainable=True)

  def __call__(self, positions):
    d = tf.linalg.norm(positions[..., None, :] -
                       self.head_centers / self.center_scale,
                       axis=-1)
    weights = tf.nn.softmax(-self.temperature * d)
    return weights


class GridDecomposition(tf.Module):

  def __init__(self, scene_bbox, n_heads):
    self.n_heads = n_heads

    dim = int(np.ceil(n_heads**(1 / 3)))
    i = tf.linspace(0.0, 1.0, dim + 2)[1:-1]
    xyz = tf.reshape(tf.stack(tf.meshgrid(i, i, i), axis=-1), (-1, 3))

    self.center_scale = 1.0
    self.head_centers = xyz * (scene_bbox[1] - scene_bbox[0]) + scene_bbox[0]

  def __call__(self, positions):
    d = tf.linalg.norm(positions[..., None, :] - self.head_centers, axis=-1)
    weights = tf.one_hot(tf.argmax(-d, axis=-1), self.n_heads)
    return weights


class DecomposedRadianceField(RadianceFieldBase):

  def __init__(self,
               decomposition_model,
               head_constructor=lambda: RadianceField(64)):
    self.decomposition_model = decomposition_model
    self.n_heads = self.decomposition_model.n_heads
    self.pilot = head_constructor()
    self.coarse = head_constructor()
    self.using_pilot = True
    self.heads = list(head_constructor() for _ in range(self.n_heads))

  def get_decomposition_vars(self):
    return self.decomposition_model.trainable_variables

  def get_radiance_vars(self):
    rvars = []
    rvars += self.coarse.trainable_variables
    rvars += self.pilot.trainable_variables
    for head in self.heads:
      rvars += head.trainable_variables

    return rvars

  def trace_rays_importance(self,
                            rays,
                            samples,
                            samples_coarse,
                            deterministic=False,
                            **kwargs):
    if self.using_pilot:
      return self.trace_rays(rays,
                             samples,
                             deterministic=deterministic,
                             **kwargs)

    else:
      coarse_xyz, coarse_dir, _, _, pdfx = self.gen_ray_samples(
          rays, samples_coarse, deterministic=deterministic)
      _, pdfy, _ = self.coarse.sample_field(coarse_xyz,
                                            coarse_dir,
                                            density_noise=0.0)

      pdf = tf.stack([pdfx, pdfy], axis=-1)
      return self.trace_rays(rays,
                             samples,
                             pdf=pdf,
                             deterministic=deterministic,
                             **kwargs)

  def render_importance(self,
                        pose,
                        farplane,
                        focal,
                        principal,
                        res,
                        samples,
                        samples_coarse,
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
      rgba_i, depth_i, attention_i = self.trace_rays_importance(
          rays[i * chunk:(i + 1) * chunk],
          samples=samples,
          samples_coarse=samples_coarse,
          deterministic=False,
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

  def sample_field(self, positions, directions, density_noise=1e0, **kwargs):
    if "head_index" in kwargs:
      i = kwargs["head_index"]
      del kwargs["head_index"]
      return self.sample_head(i, positions, directions, density_noise, **kwargs)

    if "use_coarse" in kwargs and kwargs["use_coarse"]:
      decomposition = self.decomposition_model(positions)
      radiance, density, _ = self.coarse.sample_field(positions, directions,
                                                      density_noise, **kwargs)

    if self.using_pilot:
      decomposition = self.decomposition_model(positions)
      radiance, density, _ = self.pilot.sample_field(positions, directions,
                                                     density_noise, **kwargs)

    else:
      decomposition = self.decomposition_model(positions)
      if self.coarse is not None:
        decomposition = tf.stop_gradient(decomposition)

      density = 0.0
      radiance = 0.0

      for i, head in enumerate(self.heads):
        w_i = decomposition[..., i]

        radiance_i, density_i, _ = head.sample_field(positions, directions,
                                                     density_noise, **kwargs)

        density += density_i * w_i
        radiance += radiance_i * w_i[..., None]

    return radiance, density, decomposition

  def sample_head(self, i, positions, directions, density_noise=1e0, **kwargs):
    if self.using_pilot:
      radiance, density, _ = self.pilot.sample_field(positions, directions,
                                                     density_noise, **kwargs)

    else:
      radiance, density, _ = self.heads[i].sample_field(positions, directions,
                                                        density_noise, **kwargs)

    decomposition = tf.ones_like(radiance[..., :1])
    return radiance, density, decomposition


def get_im_rays(pose, farplane, focal, principal, res):
  i = tf.range(res[0])
  j = tf.range(res[1])
  ij = tf.reshape(tf.stack(tf.meshgrid(i, j, indexing="ij"), axis=-1), (-1, 2))
  xy = tf.reshape(tf.cast(ij, tf.float32), (-1, 2))[:, ::-1]
  ip_points = perspective.ray(xy, focal, principal)
  ws_ip_points = apply_transform(pose, ip_points)
  ws_eye = apply_transform(pose, tf.zeros((1, 3)))
  eyes_tiled = tf.tile(ws_eye, (res[0] * res[1], 1))
  ray_dirs, _ = tf.linalg.normalize(ws_ip_points - eyes_tiled, axis=-1)
  rays = tf.concat([eyes_tiled, farplane * ray_dirs], axis=-1)

  return rays


def get_ray_cell_bounds(model, rays):
  if model.n_heads == 1:
    nr = tf.shape(rays)[0]
    bounds = tf.stack([tf.zeros((nr, 1)), tf.ones((nr, 1))], axis=-1)
    mask = tf.ones((nr, 1), dtype=tf.bool)
    return bounds, mask

  o = rays[:, None, None, :3]
  d = rays[:, None, None, 3:]

  def get_delaunay(sites):
    triangulation = Delaunay(sites)
    return triangulation.vertex_neighbor_vertices

  p = model.decomposition_model.head_centers / model.decomposition_model.center_scale
  row_starts, neighbour_inds = tf.py_function(get_delaunay, [p],
                                              (tf.int64, tf.int64))
  row_starts = tf.reshape(row_starts, (-1,))
  neighbour_inds = tf.reshape(neighbour_inds, (-1,))

  p1_vals = tf.gather(p, neighbour_inds)
  p1 = tf.RaggedTensor.from_row_starts(values=p1_vals,
                                       row_starts=row_starts[:-1])
  p0 = tf.ones_like(p1) * p[:, None]

  p0 = p0[None]
  p1 = p1[None]

  def rdot(a, b):
    prod = a * b
    return prod[..., 0] + prod[..., 1] + prod[..., 2]

  pm = (p0 + p1) / 2
  t = rdot(pm - o, p1 - p0) / rdot(d, p1 - p0)

  backfacing = tf.cast(tf.less(rdot(p1 - p0, d), 0.0), tf.float32)

  back_t = t + backfacing * (1e10 * tf.ones_like(t))
  front_t = t + (1.0 - backfacing) * (-1e10 * tf.ones_like(t))

  head_max = tf.reduce_min(back_t, axis=2).to_tensor()
  head_min = tf.reduce_max(front_t, axis=2).to_tensor()

  head_max = tf.clip_by_value(head_max, 0.0, 1.0)
  head_min = tf.clip_by_value(head_min, 0.0, 1.0)

  bounds = tf.stack([head_min, head_max], axis=-1)
  mask = tf.greater(head_max, head_min)

  return bounds, mask


def fast_voronoi_render(model,
                        pose,
                        farplane,
                        focal,
                        principal,
                        res,
                        samples,
                        chunk=64,
                        return_invocations=False,
                        return_layers=False):

  with tf.device("CPU"):
    rays = get_im_rays(pose, farplane, focal, principal, res)
    rcell_bounds, rcell_masks = get_ray_cell_bounds(model, rays)
    ray_samples = tf.cast(
        (rcell_bounds[..., 1] - rcell_bounds[..., 0]) * samples, tf.int32)
    ray_samples = tf.clip_by_value(tf.abs(ray_samples), 0, samples)

  rgba_r_layers = []
  depth_r_layers = []
  ind_r_layers = []

  total_invocations = 0

  for j in range(model.n_heads):
    inds_j = tf.where(rcell_masks[:, j])
    rays_j = tf.gather(rays, inds_j[:, 0])
    bounds_j = tf.gather(rcell_bounds[:, j], inds_j[:, 0])

    starts_j = rays_j[:, :3] + rays_j[:, 3:] * bounds_j[:, :1]
    offsets_j = rays_j[:, 3:] * (bounds_j[:, 1:] - bounds_j[:, :1])
    rays_j = tf.concat([starts_j, offsets_j], axis=-1)

    samples_j = tf.gather(ray_samples[:, j], inds_j[:, 0])
    n_rays_j = tf.shape(rays_j)[0]

    def cond(i, rgba, depth, decomposition, invocations):
      return i < (1 + n_rays_j // chunk)

    def body(i, rgba, depth, decomposition, invocations):
      samples_i = tf.reduce_max(samples_j[i * chunk:(i + 1) * chunk])
      samples_i = tf.clip_by_value(samples_i, 1, samples)
      coarse_samples_i = tf.clip_by_value(samples_i // 2, 1, samples)
      rays_i = rays_j[i * chunk:(i + 1) * chunk]
      rgba_i, depth_i, decomposition_i = model.trace_rays_importance(
          rays_i, samples_i, coarse_samples_i, deterministic=True, head_index=j)

      rgba = tf.concat([rgba, rgba_i], axis=0)
      depth = tf.concat([depth, depth_i], axis=0)
      decomposition = tf.concat([decomposition, decomposition_i], axis=0)
      invocations += samples_i * tf.shape(rays_i)[0]

      return [i + 1, rgba, depth, decomposition, invocations]

    lvars = [
        0,
        tf.zeros((0, 4)),
        tf.zeros((0, 1)),
        tf.zeros((0, model.heads[j].n_heads)), 0
    ]
    _, rgba_j, depth_j, _, invocations_j = tf.while_loop(
        cond,
        body,
        lvars,
        parallel_iterations=1,
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, 4]),
            tf.TensorShape([None, 1]),
            tf.TensorShape([None, model.heads[j].n_heads]),
            tf.TensorShape([])
        ])

    with tf.device("CPU"):
      rgba_r_layers.append(tf.identity(rgba_j))
      depth_r_layers.append(tf.identity(depth_j))
      ind_r_layers.append(tf.identity(inds_j))
    total_invocations += invocations_j

  with tf.device("CPU"):
    rgba_0 = tf.zeros((tf.shape(rays)[0], 4))
    depth_0 = tf.zeros((tf.shape(rays)[0], 1))

    rgba_layers = []
    depth_layers = []

    for j in range(model.n_heads):
      rgba_j = tf.tensor_scatter_nd_update(rgba_0, ind_r_layers[j],
                                           rgba_r_layers[j])
      depth_j = tf.tensor_scatter_nd_update(depth_0, ind_r_layers[j],
                                            depth_r_layers[j])

      rgba_j = tf.reshape(rgba_j, (res[0], res[1], 4))
      depth_j = tf.reshape(depth_j, (res[0], res[1], 1))

      rgba_layers.append(rgba_j)
      depth_layers.append(depth_j)

    rgba_layers = tf.stack(rgba_layers, axis=0)
    depth_layers = tf.stack(depth_layers, axis=0)

    ws_sites = model.decomposition_model.head_centers

    ws_eye = apply_transform(pose, tf.zeros((1, 3)))
    head_order = tf.argsort(tf.linalg.norm(ws_sites - ws_eye, axis=-1),
                            direction="DESCENDING")

    result_rgba = rgba_layers[head_order[0]]
    result_depth = depth_layers[head_order[0]]
    result_decomposition = tf.zeros(
        (res[0], res[1], model.n_heads)) + tf.one_hot(
            tf.constant(0)[None, None], model.n_heads)
    for i in range(1, model.n_heads):
      j = head_order[i]

      rgba = rgba_layers[j]
      alpha = rgba_layers[j, :, :, 3:]
      depth = depth_layers[j]
      decomposition = tf.one_hot(tf.constant(i)[None, None], model.n_heads)

      result_rgba = rgba + (1.0 - alpha) * result_rgba
      result_depth = depth + (1.0 - alpha) * result_depth
      result_decomposition = alpha * decomposition + (
          1.0 - alpha) * result_decomposition

  if return_invocations:
    return result_rgba, result_depth, result_decomposition, total_invocations
  elif return_layers:
    return result_rgba, result_depth, result_decomposition, rgba_layers, head_order
  else:
    return result_rgba, result_depth, result_decomposition
