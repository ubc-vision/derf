from model.nerf import *
from model.derf import *


def build_model(argv, scene_bbox):
  if argv.decomposition == "voronoi":
    decomposition = VoronoiDecomposition(scene_bbox, argv.n_heads)

  elif argv.decomposition == "network":
    decomposition = MLPDecomposition(argv.n_heads,
                                     pos_feature=PositionalEncoding(
                                         argv.attn_penc_depth))

  elif argv.decomposition == "grid":
    decomposition = GridDecomposition(scene_bbox, argv.n_heads)

  def head_constructor():
    return RadianceField(argv.n_units,
                         argv.n_layers,
                         pos_feature=PositionalEncoding(argv.penc_depth),
                         disable_view_dependence=argv.no_view_dependence)

  model = DecomposedRadianceField(decomposition, head_constructor)

  # Force all weight allocations
  _ = model.trace_rays_importance(tf.zeros((1, 6)), 10, 5)
  model.using_pilot = False
  _ = model.trace_rays_importance(tf.zeros((1, 6)), 10, 5)
  model.using_pilot = True

  return model


def restore_model(location,
                  model,
                  optimizer=None,
                  iteration_count=None,
                  require=False):
  if optimizer is None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    print(optimizer.get_weights())

  if iteration_count is None:
    iteration_count = tf.Variable(0, dtype=tf.int64)

  checkpoint = tf.train.Checkpoint(_model=model,
                                   _optimizer=optimizer,
                                   _iteration_count=iteration_count)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                  location,
                                                  max_to_keep=2)

  if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    print("Checkpoint {} loaded".format(checkpoint_manager.latest_checkpoint))
    print("Currently at iteration {}".format(iteration_count.numpy()))

  else:
    print("No checkpoint found: {}".format(location))
    if require:
      quit()

  return checkpoint_manager