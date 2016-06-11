# This is implementation of [FractalNet](http://arxiv.org/abs/1605.07648) paper

import os
import numpy as np
import time
import tensorflow as tf
from loadData import *
import tflearn
from net import FractalNet

logdir = os.path.expanduser("~/logs/mini")

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_string('train_dir', logdir, 'training dir')
flags.DEFINE_integer('train_log', 10, 'How many steps to log training')
flags.DEFINE_integer('test_log', 500, 'How many steps to log test error')

g = tf.Graph()

with g.as_default():
    with tf.name_scope('input'):
        with tf.device('/cpu:0'):
            input_images = tf.constant(train_data, dtype=tf.float32)
            input_labels = tf.constant(train_labels, dtype=tf.float32)

            test_images = tf.constant(test_data, dtype=tf.float32)
            test_labels = tf.constant(test_label, dtype=tf.float32)

            image, label = tf.train.slice_input_producer(
                [input_images, input_labels], num_epochs=FLAGS.num_epochs)
            X_batch, Y_batch = tf.train.batch(
                [image, label], batch_size=FLAGS.batch_size)


    X = tf.placeholder_with_default(X_batch, [None, 32, 32, 3])
    Y = tf.placeholder_with_default(Y_batch, [None, 10])

    FF = []
    net = X

    for i, channel_no in enumerate([32, 64, 128, 256, 512]):
        with tf.name_scope("block_%d" % (i + 1)):
            net = FractalNet(2, net, channel_no, 2)
        FF.append(net)
        net = tf.nn.max_pool(net.get_tensor(), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        print(i, net.get_shape())

    net = tflearn.fully_connected(net, noClasses)
    print(net.get_shape())
    print(Y.get_shape)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))

    yp = tf.nn.softmax(net)
    acc = tf.reduce_mean(
        tf.cast(tf.nn.in_top_k(yp, tf.argmax(Y, 1), k=1), tf.float32)
    )

    learning_rate = tf.Variable(0.01, trainable=False)
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    train_summ = tf.merge_summary([
        tf.scalar_summary("Learning rate", learning_rate),
        tf.scalar_summary("Train accuracy", acc),
        tf.scalar_summary("Train loss", loss),
    ],
        name="Training_summary"
    )

    test_summ = tf.merge_summary([
        tf.scalar_summary("Test accuracy", acc),
        tf.scalar_summary("Test loss", loss),
    ],
        name="Test_summary"
    )

with tf.Session(graph=g, config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(logdir, sess.graph)

    t = time.clock()
    try:
        step = 0
        while not coord.should_stop():
            step += 1

            # Training on
            tflearn.is_training(True)

            if step % FLAGS.train_log == 0 or step == 1:
                summ, _ = sess.run([train_summ, train])
                writer.add_summary(summ, step)
            else:
                sess.run(train)

            if step % FLAGS.test_log == 0 or step == 1:
                dt = time.clock() - t

                perm = np.random.permutation(10000 - 1)[:1000]
                tflearn.is_training(False)
                summ, top1, avg_loss = sess.run([test_summ, acc, loss],             feed_dict={
                    X: test_data[perm],
                    Y: test_label[perm],
                })

                writer.add_summary(summ, step)
                print("[{:10d}] acc: {:2.5f}%, loss: {:2.7f} steps/sec {:5.2f}"
                    .format(step, 100 * top1, avg_loss, FLAGS.test_log / dt))

                t = time.clock()

    except tf.errors.OutOfRangeError:
      print('Saving')
      saver.save(sess, FLAGS.train_dir, global_step=step)
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
        coord.request_stop()
    coord.join(threads)
