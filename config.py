import os

import configargparse

import tensorflow as tf


def print_argv(argv):
  args = list(vars(argv))
  args.sort()
  print("---------------------- Arguments ----------------------")
  print()
  for arg in args:
    print(arg.rjust(25, " ") + "  " + str(getattr(argv, arg)))
  print()
  print("-------------------------------------------------------")


def get_base_parser():
  parser = configargparse.ArgumentParser()

  parser.add_argument("experiment_name", type=str)
  parser.add_argument("dataset", type=str)

  parser.add_argument("--experiments_dir",
                      type=str,
                      default="./experiments",
                      env_var="MY_EXPERIMENTS")
  parser.add_argument("--datasets_dir",
                      type=str,
                      default="./datasets",
                      env_var="MY_DATASETS")
  parser.add_argument("--no_distribute", action="store_true")

  # Training Parameters
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=1024)

  parser.add_argument("--coarse_only_iterations", type=int, default=10**5)
  parser.add_argument("--total_iterations", type=int, default=4 * 10**5)

  # Model Parameters
  parser.add_argument("--decomposition",
                      choices=["network", "voronoi", "grid"],
                      default="voronoi")
  parser.add_argument("--no_view_dependence", action="store_true")
  parser.add_argument("--white_background", action="store_true")

  parser.add_argument("--n_samples", type=int, default=192)
  parser.add_argument("--n_coarse_samples", type=int, default=64)
  parser.add_argument("--n_heads", type=int, default=16)
  parser.add_argument("--n_units", type=int, default=256)
  parser.add_argument("--n_layers", type=int, default=8)

  parser.add_argument("--penc_depth", type=int, default=12)
  parser.add_argument("--attn_penc_depth", type=int, default=7)

  parser.add_argument("--fp16", action="store_true")

  parser.add_argument("--seperate_attention_loss", action="store_true")

  return parser


def start_script(parser, entry_point):
  argv = parser.parse_args()

  expdir = os.path.join(argv.experiments_dir, argv.experiment_name)
  conf = os.path.join(expdir, "conf.ini")
  if os.path.exists(conf):
    with open(conf, "r") as fd:
      conf_text = fd.read()
    argv = parser.parse_args(config_file_contents=conf_text)

  else:
    os.makedirs(expdir, exist_ok=True)
    parser.write_config_file(argv, [conf])

  if argv.fp16:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)

  print_argv(argv)

  entry_point(argv)