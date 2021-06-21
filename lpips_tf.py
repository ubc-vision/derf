import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf
from six.moves import urllib
import configargparse
import imageio

# safer to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.disable_eager_execution()

_URL = 'http://rail.eecs.berkeley.edu/models/lpips'


def _download(url, output_dir):
    """Downloads the `url` file into `output_dir`.
    Modified from https://github.com/tensorflow/models/blob/master/research/slim/datasets/dataset_utils.py
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def lpips(input0, input1, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.
    Args:
        input0: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].
        input1: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].
    Returns:
        The Learned Perceptual Image Patch Similarity (LPIPS) distance.
    Reference:
        Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018.
    """
    # flatten the leading dimensions
    batch_shape = tf.shape(input0)[:-3]
    input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
    input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
    # NHWC to NCHW
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])
    # normalize to [-1, 1]
    input0 = input0 * 2.0 - 1.0
    input1 = input1 * 2.0 - 1.0

    input0_name, input1_name = '0:0', '1:0'

    default_graph = tf.get_default_graph()
    producer_version = default_graph.graph_def_versions.producer

    cache_dir = os.path.expanduser('~/.lpips')
    os.makedirs(cache_dir, exist_ok=True)
    # files to try. try a specific producer version, but fallback to the version-less version (latest).
    pb_fnames = [
        '%s_%s_v%s_%d.pb' % (model, net, version, producer_version),
        '%s_%s_v%s.pb' % (model, net, version),
    ]
    for pb_fname in pb_fnames:
        if not os.path.isfile(os.path.join(cache_dir, pb_fname)):
            try:
                _download(os.path.join(_URL, pb_fname), cache_dir)
            except:
                pass
        if os.path.isfile(os.path.join(cache_dir, pb_fname)):
            break

    with open(os.path.join(cache_dir, pb_fname), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,
                                input_map={input0_name: input0, input1_name: input1})
        distance, = default_graph.get_operations()[-1].outputs

    if distance.shape.ndims == 4:
        distance = tf.squeeze(distance, axis=[-3, -2, -1])
    # reshape the leading dimensions
    distance = tf.reshape(distance, batch_shape)
    return distance


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("image_dir", type=str)
    argv = parser.parse_args()

    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)
    # https://github.com/bmild/nerf/issues/66
    distance_t = lpips(image0_ph, image1_ph, model='net-lin', net='vgg')
    out = []
    for cur, dirs, files in os.walk(argv.image_dir):
        for file in sorted(files):
            if file.startswith("gt_"):
                gt_img_path = os.path.join(cur, file)
                pred_img_path = os.path.join(cur, file.replace("gt", "pred"))
                gt_img = imageio.imread(gt_img_path, pilmode='RGB') / 255.0
                pred_img = imageio.imread(pred_img_path, pilmode='RGB') / 255.0
                with tf.Session() as session:
                    distance = session.run(distance_t, feed_dict={image0_ph: gt_img, image1_ph: pred_img})
                out.append(distance)
    print(np.array(out).astype(np.float32).tobytes().hex())
