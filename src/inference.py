"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 128
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('direction', 'A->B', 'A->B')
tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', 'input_sample.jpg', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '128', 'image size, default: 128')

def inference():
  graph = tf.Graph()
  # visualization staff yw3025
  index = open(FLAGS.input + "/" + FLAGS.direction + "index.html", "w")
  index.write("<html><body><table><tr>")
  index.write("<th>name</th><th>input</th><th>output</th></tr>")
  # batch inference staff yw3025
  for file in os.listdir(FLAGS.input):
      filename = FLAGS.input + "/" + file
      print(filename)
      with graph.as_default():
        with tf.gfile.FastGFile(filename, 'rb') as f:
          image_data = f.read()
          input_image = tf.image.decode_jpeg(image_data, channels=3)
          input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
          input_image = utils.convert2float(input_image)
          input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

        with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(model_file.read())
        [output_image] = tf.import_graph_def(graph_def,
                              input_map={'input_image': input_image},
                              return_elements=['output_image:0'],
                              name='output')

      with tf.Session(graph=graph) as sess:
        generated = output_image.eval()
        with open(FLAGS.output + "/" + FLAGS.direction + file, 'wb') as f:
          f.write(generated)
      # visualization staff yw3025
      index.write("<td>%s</td>" % filename)
      index.write("<td><img src='%s'></td>" % file)
      index.write("<td><img src='%s'></td>" % (FLAGS.direction + file))
      index.write("</tr>")

      print("processing" + filename)

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
