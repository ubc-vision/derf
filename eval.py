import json
import os
import subprocess
import time

import numpy as np
import tensorflow as tf
import tqdm

from config import get_base_parser, start_script
from builder import build_model, restore_model
from data_io.loader import load_scene
from data_io.rays_dataset import rays_dataset
from model.derf import fast_voronoi_render
from utils.misc import weights_to_rgb, imwrite


def eval(argv):
  experiment_dir = os.path.join(argv.experiments_dir, argv.experiment_name)

  ims_ds, near, far, focal, principal = load_scene(argv, argv.split)

  _, scene_bbox = rays_dataset(ims_ds, near, far, focal, principal,
                               argv.batch_size)

  model = build_model(argv, scene_bbox)

  checkpoint_location = os.path.join(experiment_dir, "ckpt")

  iteration_count = tf.Variable(0, dtype=tf.int64)
  _ = restore_model(checkpoint_location,
                    model,
                    iteration_count=iteration_count,
                    require=True)
  model.using_pilot = (iteration_count.numpy() < argv.coarse_only_iterations)

  gt_rgbs = []
  pred_rgbs = []
  pred_depths = []
  pred_decs = []
  psnrs = []
  ssims = []
  invocation_counts = []
  runtimes = []

  _, rgba = next(e for e in ims_ds.take(1))

  dims = tf.cast(rgba.shape[:2], tf.float32)
  sf = argv.img_scale
  dims = tf.cast(sf * dims, tf.int32)
  scale_transform = tf.linalg.diag([1. / far, 1. / far, 1. / far, 1.0])

  chunk = (2**22) // (argv.n_units**2)

  if argv.decomposition == "network":

    @tf.function
    def render(pose):
      rres = model.render(pose,
                          1.5,
                          focal * sf,
                          principal * sf,
                          dims,
                          samples=argv.n_samples,
                          chunk=chunk)
      return (*rres, 1)

  else:

    @tf.function
    def render(pose):
      return fast_voronoi_render(model,
                                 pose,
                                 1.5,
                                 focal * sf,
                                 principal * sf,
                                 dims,
                                 samples=argv.n_samples,
                                 chunk=chunk,
                                 return_invocations=True)

  pose0 = list(ims_ds.take(1))[0][0]
  render = render.get_concrete_function(pose0)
  _ = render(pose0)

  dummy = 0

  for pose, rgba in tqdm.tqdm(ims_ds):
    pose = scale_transform @ pose
    rgba = tf.image.resize(rgba, dims)

    start = time.perf_counter()

    pred_rgba, pred_depth, pred_dec, invocations = render(pose)

    dummy += pred_rgba[0, 0, 0]
    runtimes.append(float(time.perf_counter() - start))
    invocation_counts.append(int(invocations))

    gt_rgbs.append(rgba[..., :3].numpy())
    pred_rgbs.append(pred_rgba[..., :3].numpy())
    pred_depths.append(pred_depth.numpy())
    pred_decs.append(pred_dec.numpy())

    # https://github.com/bmild/nerf/issues/66
    psnrs.append(float(tf.image.psnr(rgba[..., :3],
                                     pred_rgba[..., :3],
                                     max_val=1.0)))
    ssims.append(float(tf.image.ssim(rgba[..., :3],
                                     pred_rgba[..., :3],
                                     max_val=1.0)))

  os.makedirs(argv.out_dir, exist_ok=True)

  for i in range(len(pred_rgbs)):
    imwrite(os.path.join(argv.out_dir, "pred_rgb_{}.png".format(i)),
            pred_rgbs[i])
    imwrite(os.path.join(argv.out_dir, "gt_rgb_{}.png".format(i)), gt_rgbs[i])

    imwrite(os.path.join(argv.out_dir, "decomp_{}.png".format(i)),
            weights_to_rgb(pred_decs[i]))

  lpips_res = subprocess.run(["python", "./lpips_tf.py", argv.out_dir],
                             stdout=subprocess.PIPE)
  hexval = lpips_res.stdout.decode().strip()
  lpipss = list(
      float(x) for x in np.frombuffer(bytearray.fromhex(hexval), np.float32))

  summary_data = {
      "name": argv.experiment_name,
      "psnrs": psnrs,
      "ssims": ssims,
      "lpipss": lpipss,
      "runtimes": runtimes,
      "invocations": invocation_counts
  }
  with open(os.path.join(argv.out_dir, argv.experiment_name + ".json"),
            "w") as f:
    json.dump(summary_data, f)


if __name__ == "__main__":
  parser = get_base_parser()
  parser.add_argument("out_dir", type=str)
  parser.add_argument("--split", type=str, default="test")
  parser.add_argument("--img_scale", type=float, default=1.0)
  start_script(parser, eval)
