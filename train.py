import os

from matplotlib import cm
import tensorflow as tf
import tqdm

from config import get_base_parser, start_script
from builder import build_model, restore_model
from data_io.loader import load_scene
from data_io.rays_dataset import rays_dataset
from utils.misc import weights_to_rgb


def train(argv):
  experiment_dir = os.path.join(argv.experiments_dir, argv.experiment_name)

  if argv.no_distribute:
    distribute_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
  else:
    distribute_strategy = tf.distribute.MirroredStrategy()

  with distribute_strategy.scope():
    ims_ds, near, far, focal, principal = load_scene(argv, "train")

    rays_ds, scene_bbox = rays_dataset(ims_ds, near, far, focal, principal,
                                       argv.batch_size)
    rays_ds = distribute_strategy.experimental_distribute_dataset(rays_ds)

    model = build_model(argv, scene_bbox)

    checkpoint_location = os.path.join(experiment_dir, "ckpt")
    os.makedirs(checkpoint_location, exist_ok=True)

    learning_rate = tf.Variable(1e-4)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if argv.fp16:
      optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          optimizer, "dynamic")

    iteration_count = tf.Variable(0, dtype=tf.int64)

    checkpoint_manager = restore_model(checkpoint_location, model, optimizer,
                                       iteration_count)

  summary_location = os.path.join(experiment_dir, "logs")

  os.makedirs(summary_location, exist_ok=True)
  summary_writer = tf.summary.create_file_writer(summary_location)
  with summary_writer.as_default():
    with open(os.path.join(experiment_dir, "conf.ini")) as fd:
      conf_text = fd.read().replace("\n", "\n\n")

    tf.summary.text("Arguments", conf_text, step=0)

  summary_pose, summary_gt_rgb = list(ims_ds.take(1))[0]

  def gen_summary():
    scale_transform = tf.linalg.diag([1. / far, 1. / far, 1. / far, 1.0])

    if argv.n_coarse_samples > 0:
      rgb, depth, decomposition_im = model.render_importance(
          scale_transform @ summary_pose,
          1.5, [200.0] * 2,
          tf.constant([128., 128.]), [256, 256],
          samples=argv.n_samples,
          samples_coarse=argv.n_coarse_samples)

    else:
      rgb, depth, decomposition_im = model.render(
          scale_transform @ summary_pose,
          1.5, [200.0] * 2,
          tf.constant([128., 128.]), [256, 256],
          samples=argv.n_samples)

    decomposition_rgb = weights_to_rgb(decomposition_im)

    return rgb, depth, decomposition_rgb

  n_replicas = distribute_strategy.num_replicas_in_sync

  def step(ray, color_gt):

    def pred_fn():
      if argv.n_coarse_samples > 0:
        color_pred, _, decomposition_pred = model.trace_rays_importance(
            ray, samples=argv.n_samples, samples_coarse=argv.n_coarse_samples)

        if model.using_pilot:
          color_coarse_pred = 0.0 * color_pred

        else:
          color_coarse_pred, _, _ = model.trace_rays(ray, argv.n_coarse_samples)

      else:
        color_pred, _, decomposition_pred = model.trace_rays(
            ray, samples=argv.n_samples)

        color_coarse_pred = 0.0 * color_pred

      return color_pred, decomposition_pred, color_coarse_pred

    radiance_loss = tf.constant(0.0)
    decomposition_loss = tf.constant(0.0)

    def rad_loss_fn():
      nonlocal radiance_loss
      color_pred, _, color_coarse_pred = pred_fn()

      alpha = color_pred[..., 3:]
      alpha_coarse = color_coarse_pred[..., 3:]
      if argv.white_background:
        color_pred = alpha * color_pred[..., :3] + (1.0 - alpha)
        color_coarse_pred = alpha_coarse * color_coarse_pred[..., :3] + (
            1.0 - alpha_coarse)

      else:
        color_pred = alpha * color_pred[..., :3]
        color_coarse_pred = alpha_coarse * color_coarse_pred[..., :3]

      color_gt_rgb = color_gt[..., :3]

      radiance_loss = tf.reduce_mean((color_gt_rgb - color_pred)**2)
      radiance_loss += tf.reduce_mean((color_gt_rgb - color_coarse_pred)**2)
      radiance_loss /= n_replicas
      return radiance_loss

    def dec_loss_fn():
      nonlocal decomposition_loss
      _, decomposition_pred, _ = pred_fn()
      dc_pred_weighted = decomposition_pred * color_gt[..., 3:]

      decomposition_loss = tf.reduce_mean(
          tf.reduce_mean(dc_pred_weighted, axis=0)**2)

      if argv.decomposition == "network":
        # Sparsity for MLP
        decomposition_loss += 0.1 * tf.reduce_mean(
            (dc_pred_weighted + 1e-3)**0.5)

      decomposition_loss /= n_replicas
      return decomposition_loss

    decomposition_vars = model.get_decomposition_vars()
    if model.using_pilot:
      optimizer.minimize(dec_loss_fn, decomposition_vars)

    radiance_vars = model.get_radiance_vars()
    optimizer.minimize(rad_loss_fn, radiance_vars)

    res = {}
    res["Radiance Loss"] = radiance_loss
    res["Decomposition Loss"] = decomposition_loss

    return res

  def distributed_step(*args):
    replica_losses = distribute_strategy.run(step, args=args)
    mean_losses = {}
    for key in replica_losses:
      mean_losses[key] = distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                    replica_losses[key],
                                                    axis=None)

    return mean_losses

  def compile():

    @tf.function
    def step(*args):
      return distributed_step(*args)

    @tf.function
    def summary():
      return gen_summary()

    return step, summary

  compiled_step, compiled_summ = compile()

  losses = []
  for ray, color_gt in tqdm.tqdm(rays_ds):
    if iteration_count.numpy() >= argv.total_iterations:
      break

    if iteration_count.numpy() >= argv.coarse_only_iterations:
      if model.using_pilot:
        model.using_pilot = False
        compiled_step, compiled_summ = compile()

    if argv.decomposition == "voronoi":
      t = tf.cast(iteration_count, tf.float32) / float(
          argv.coarse_only_iterations)
      model.decomposition_model.temperature.assign(
          10.0**(9 * tf.clip_by_value(t, 0.0, 1.0)))

    lrt = tf.clip_by_value(
        tf.cast(iteration_count - argv.coarse_only_iterations, tf.float32) /
        (argv.total_iterations - argv.coarse_only_iterations), 0.0, 1.0)
    learning_rate.assign(5.0 * 10**(-(4 + lrt)))

    losses.append(compiled_step(ray, color_gt))

    if (iteration_count.numpy() % 5000) == 0:
      rgb, depth, decomposition_im = compiled_summ()

      with summary_writer.as_default():
        tf.summary.image("RGB", rgb[None], step=iteration_count)
        if iteration_count.numpy() == 0:
          tf.summary.image("RGB Ground Truth",
                           summary_gt_rgb[None],
                           step=iteration_count)

        tf.summary.image("Pixel Decomposition",
                         decomposition_im[None],
                         step=iteration_count)

        cmap = cm.ScalarMappable(cmap=cm.get_cmap("hot"))
        depth_color = cmap.to_rgba(depth[:, :, 0], norm=True)[None]
        tf.summary.image("Depth", depth_color, step=iteration_count)

        for key in losses[0]:
          avg_loss = tf.reduce_mean(list(li[key] for li in losses))
          tf.summary.scalar(key, avg_loss, step=iteration_count)

        losses = []

      summary_writer.flush()

    if (iteration_count.numpy() % 10000) == 0:
      checkpoint_manager.save()

    iteration_count.assign_add(1)


if __name__ == "__main__":
  start_script(get_base_parser(), train)