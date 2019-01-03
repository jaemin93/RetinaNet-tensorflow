from slim_datasets import convert_tf_record
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_name',
    'mnist',
    'The name of the dataset prefix.')
tf.app.flags.DEFINE_string(
    'dataset_dir',
    './raw_data/mnist',
    'A directory containing a set of subdirectories representing class names. Each subdirectory should contain PNG or JPG encoded images.')
tf.app.flags.DEFINE_integer(
    'num_shards',
    5,
    'A number of sharding for TFRecord files(integer).')
tf.app.flags.DEFINE_float(
    'ratio_val',
    0.2,
    'A ratio of validation datasets for TFRecord files(flaot, 0 ~ 1).')
def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  convert_tf_record.run(FLAGS.dataset_name, FLAGS.dataset_dir, FLAGS.num_shards, FLAGS.ratio_val)
if __name__ == '__main__':
  tf.app.run()