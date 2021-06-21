import os

from data_io.deepvoxel_data import deepvoxel_dataset
from data_io.nerf_real_data import nerf_real_dataset
from data_io.nerf_synth_data import nerf_synth_dataset


def load_scene(argv, split):
  if argv.dataset[:2] == "dv":
    return deepvoxel_dataset(os.path.join(argv.datasets_dir, "deepvoxels",
                                          argv.dataset[3:]),
                             split=split)

  elif argv.dataset[:4] == "llff":
    return nerf_real_dataset(os.path.join(argv.datasets_dir, "nerf_llff_data",
                                          argv.dataset[5:]),
                             split=split,
                             scale=1 / 4)

  elif argv.dataset[:4] == "nerf":
    return nerf_synth_dataset(os.path.join(argv.datasets_dir, "nerf_synthetic",
                                           argv.dataset[5:]),
                              split=split)

  else:
    raise NotImplementedError(argv.dataset)