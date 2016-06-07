# This is implementation of [FractalNet](http://arxiv.org/abs/1605.07648) paper

import os
import numpy as np
import time
import tensorflow as tf
from loadData import *
import tflearn
from net import FractalNet

batch_size = 50
logdir = os.path.expanduser("~/logs/")
test_log_steps = 10
reduce_learning = [100, 150, 175, 185]

g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, height, width, channels], name="input")
    Y = tf.placeholder(tf.float32, [None, noClasses], name="labels")
    FF = []
    net = X

    for i, channel_no in enumerate([16, 32, 64, 128, 128]):
        with tf.name_scope("block_%d" % (i + 1)):
            net = FractalNet(net, 1, out_chanell=channel_no)
        FF.append(net)
        net = tf.nn.max_pool(net.get_tensor(), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        print(i, net.get_shape())

    net = tflearn.fully_connected(net, noClasses)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))

    yp = tf.nn.softmax(net)
    acc = tf.reduce_mean(
        tf.cast(tf.nn.in_top_k(yp, tf.argmax(Y, 1), k=1), tf.float32)
    )

    learning_rate = tf.Variable(0.01, trainable=False)
    reduce_lr = learning_rate.assign(tf.mul(learning_rate, 0.5))
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    learning_rate_summ = tf.scalar_summary("Learning rate", learning_rate)
    train_acc_summ = tf.scalar_summary("Train accuracy", acc)
    train_loss_summ = tf.scalar_summary("Train loss", loss)

    train_summ = tf.merge_summary(
        [learning_rate_summ, train_acc_summ, train_loss_summ],
        name="Training_summary"
    )

    test_acc_summ = tf.scalar_summary("Test accuracy", acc)
    test_loss_summ = tf.scalar_summary("Test loss", loss)

    test_summ = tf.merge_summary(
        [test_acc_summ, test_loss_summ],
        name="Test_summary"
    )

with tf.Session(graph=g) as sess:
    sess.run(tf.initialize_all_variables())

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(logdir, sess.graph)
    step = 0

    for epoh in range(400):
        t = time.clock()
        if epoh in reduce_learning:
            sess.run(halve_lr)

        tflearn.is_training(True)
        batch_idxs = np.random.permutation(train_data.shape[0])
        for batch in np.split(batch_idxs, train_data.shape[0] / batch_size):
            step += 1
            for F in FF[::2]:
                sess.run(F.genLocalDropPath(0.15))
            for F in FF[1::2]:
                sess.run(F.genRandomColumn())
            summ, _ = sess.run([train_summ, train], feed_dict={
                X: train_data[batch],
                Y: train_labels[batch]
            })
            writer.add_summary(summ, step)
            if step % test_log_steps == 0 or step == 1:
                dt = time.clock() - t

                tflearn.is_training(False)
                for F in FF:
                    sess.run(F.genTestMode())
                summ, top1, avg_loss = sess.run([test_summ, acc, loss], feed_dict={
                    X: test_data,
                    Y: test_label
                })

                writer.add_summary(summ, step)
                print("[{:10d}] step, acc: {:2.5f}%, loss: {:2.7f} steps/sec{:5.2f}"
                    .format(step, 100 * top1, avg_loss, test_log_steps / dt))

                t = time.clock()
