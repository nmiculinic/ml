import numpy as np
import tensorflow as tf
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_2d
from tflearn.layers.core import dropout

class FractalNet():

    def __init__(self, n, input, nb_filter, filter_size, path_keep_drop=1.0):
        self.n = n
        self.children = []
        with tf.name_scope("F%d" % n):
            with tf.name_scope("atom"):

                atom = conv_2d(input, nb_filter, filter_size, bias=False)
                atom = batch_normalization(atom)
                atom = tf.nn.relu(atom)
                atom = dropout(atom, 0.5)

            self.__tensors = [atom]
            if n > 1:
                Fp = FractalNet(n - 1, input, nb_filter, filter_size)
                self.children.append(Fp)
                Fp = FractalNet(n - 1, input, nb_filter, filter_size)
                self.children.append(Fp)
                self.__tensors.extend(Fp.__tensors)

                with tf.name_scope("join"):
                    # activations in join layer
                    # for mean join layer they should be equal and sum to 1

                    join = tf.pack(self.__tensors)
                    n, dim, h, w, _ = join.get_shape()
                    join = tf.reduce_mean(join, [0])

                    print(join.get_shape())
                    self.__tensor = join
            else:
                self.__tensor = self.__tensors[0]

    def get_tensor(self):
        return self.__tensor
