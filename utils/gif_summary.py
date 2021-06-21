# Credit to @danijar and @alexlee-gk on github
# https://github.com/tensorflow/tensorboard/issues/39#issuecomment-568917607

import numpy as np
import tensorflow as tf


def video_summary(name, video, step=None, fps=20):
  name = tf.constant(name).numpy().decode('utf-8')
  video = np.array(video)
  if video.dtype in (np.float32, np.float64):
    video = np.clip(255 * video, 0, 255).astype(np.uint8)
  B, T, H, W, C = video.shape
  try:
    frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    summary = tf.compat.v1.Summary()
    image = tf.compat.v1.Summary.Image(height=B * H, width=T * W, colorspace=C)
    image.encoded_image_string = encode_gif(frames, fps)
    summary.value.add(tag=name + '/gif', image=image)
    tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
  except (IOError, OSError) as e:
    print('GIF summaries require ffmpeg in $PATH.', e)
    frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name + '/grid', frames, step)


def encode_gif(frames, fps):
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      f'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'
  ])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out