# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import sys
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K
import tensorflow as tf


class SparseAttention(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get("uniform")
        self.supports_masking = True
        self.return_attention = return_attention
        super(SparseAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(
            shape=(input_shape[2], 1),
            name="{}_W".format(self.name),
            initializer=self.init,
        )
        self.trainable_weights = [self.W]
        self.att_weights = None
        super(SparseAttention, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))

        att_weights = tf.contrib.sparsemax.sparsemax(logits)
        self.att_weights = att_weights

        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
