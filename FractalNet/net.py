import numpy as np
import tensorflow as tf
from tflearn.layers.normalization import batch_normalization


class FractalNet():

    def __init__(self, input, n, f_height=2, f_width=2, out_chanell=None):
        _, _, _, c = input.get_shape()
        in_channel = int(c)
        if out_chanell is None:
            out_chanell = in_channel

        self.n = n
        self.children = []
        with tf.name_scope("F%d" % n):
            with tf.name_scope("atom"):
                # single comutational "atom" in FractalNet

                self.filter = tf.Variable(tf.truncated_normal(
                    [f_height, f_width, in_channel, out_chanell],
                    stddev=0.35
                ), name="filter")

                self.bias = tf.Variable([0] * out_chanell, dtype=tf.float32, name="bias")
                atom = tf.nn.conv2d(input, self.filter, [1, 1, 1, 1], 'SAME')
                atom = batch_normalization(atom)
                atom = tf.nn.relu(tf.nn.bias_add(atom, self.bias))

            self.__tensors = [atom]
            if n > 1:
                Fp = FractalNet(input, n - 1, f_height, f_width, int((out_chanell + in_channel) / 2))
                self.children.append(Fp)
                Fp = FractalNet(Fp.get_tensor(), n - 1, f_height, f_width, out_chanell)
                self.children.append(Fp)
                self.__tensors.extend(Fp.__tensors)

            with tf.name_scope("join"):
                # activations in join layer
                # for mean join layer they should be equal and sum to 1
                self.is_active = [
                    tf.Variable(1.0 / n, trainable=False, name="a%d" % i)
                    for i in range(n)
                ]

                self.__tensor = tf.add_n(
                    [tf.mul(m, x) for m, x in zip(self.is_active, self.__tensors)],
                    name="Average_pool_join"
                )

    def get_tensor(self):
        return self.__tensor
