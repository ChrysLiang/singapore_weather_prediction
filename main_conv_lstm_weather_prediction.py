
import os.path
import time

import numpy as np
import tensorflow as tf

import layer_def as ld
import BasicConvLSTMCell

from data import weather, utilities

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 60,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 48,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .0001,
                            """for learning rate""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")

def network(inputs, hidden, lstm=True):
  """
  TODO:
  1. Pooling
  2. Filter size 3
  3. Add 3 lstm
  4. Add time and date channel
  """
  conv1 = ld.conv_layer(inputs, 1, 1, 8, "encode_1", True)
  # conv2
  conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2", True)
  # conv3
  # conv3 = ld.conv_layer(conv2, 2, 1, 32, "encode_3")
  # conv4
  # conv4 = ld.conv_layer(conv3, 2, 1, 64, "encode_4")

  # y_0 = conv4
  y_0 = inputs 

  if lstm:
    # conv lstm cell
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      #cell = BasicConvLSTMCell.BasicConvLSTMCell([6,5], [3,3], 4)
      # cell = BasicConvLSTMCell.BasicConvLSTMCell([2, 3], [2, 2], 8)
      # cell = BasicConvLSTMCell.BasicConvLSTMCell([4, 5], [2, 2], 64)
      cell = BasicConvLSTMCell.BasicConvLSTMCell([4, 5], [2, 2], 8)
      if hidden is None:
        hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
      y_1, hidden = cell(y_0, hidden)
  ##  else:
  ##    y_1 = ld.conv_layer(y_0, 3, 1, 8, "encode_3")

  # conv5
  conv5 = ld.conv_layer(y_1, 3, 1, 8, "decode_5", True)

  # conv6
  # conv6 = ld.conv_layer(conv5, 2, 1, 16, "decode_6")

  # conv7
  ## conv7 = ld.conv_layer(conv6, 2, 1, 1, "decode_7")

  ##  x_1
  ##  x_1 = ld.conv_layer(conv7, 3, 2, 1, "decode_8", True) # set activation to linear
  ##  x_1 = ld.conv_layer(conv6, 2, 1, 1, "decode_7") # this might be better
  # x_1 = ld.conv_layer(conv5, 3, 1, 1, "decode_7", True)  # set activation to linear
  x_1 = ld.conv_layer(conv5, 3, 1, 1, "decode_7")  # set activation to linear
  # x_1 = y_1
  return x_1, hidden

# make a template for reuse
network_template = tf.make_template('network', network)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs : made change
    # x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 11, 9, 1])
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 4, 5, 1])
    # y = tf.placeholder(tf.float32, [None, 1, 4, 5, 1])
    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []

    # conv network
    hidden = None
    for i in range(FLAGS.seq_length - 1):
      if i < FLAGS.seq_start:
        x_1, hidden = network_template(x_dropout[:,i,:,:,:], hidden)
      else:
        x_1, hidden = network_template(x_1, hidden)
      x_unwrap.append(x_1)

    # pack them all together
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])


    # calc total loss (compare x_t to x_t+1)
    target = x[:,FLAGS.seq_start+1:,:,:,:] 
    prediction = x_unwrap[:,FLAGS.seq_start:,:,:,:]
    loss = tf.nn.l2_loss(x[:,FLAGS.seq_start+1:,:,:,:] - x_unwrap[:,FLAGS.seq_start:,:,:,:])
    # prediction = tf.reshape(x_unwrap[:, FLAGS.seq_length-1, :, :, :], [-1, 1, 4, 5, 1])
    # loss = tf.nn.l2_loss(y[:, 0, :, :, :] - tf.reshape(x_unwrap[:, FLAGS.seq_length-1, :, :, :], [-1, 1, 4, 5, 1]))
    # loss = tf.nn.l2_loss(y - prediction)
    tf.summary.scalar('loss', loss)

    # data generator
    data_generator = utilities.infinite_generator(weather.get_train(), FLAGS.batch_size)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())

    # Summary op
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

    for step in range(FLAGS.max_step):
      dat, lbl = next(data_generator)
      t = time.time()
      # current_prediction = sess.run(prediction, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      # _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, y:lbl, keep_prob:FLAGS.keep_prob})
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})

      elapsed = time.time() - t

      # print("goto training step " + str(step))

      if step % 100 == 0 and step != 0:
        # summary_str = sess.run(summary_op, feed_dict={x:dat, y:lbl, keep_prob:FLAGS.keep_prob})
        summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step)
        print("time per batch is " + str(elapsed))
        print("goto training step " + str(step))
        print(loss_r)

      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step % 1000== 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        print("saved to " + FLAGS.train_dir)




def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


