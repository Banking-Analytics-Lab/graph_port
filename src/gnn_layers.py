import tensorflow as tf
import tensorflow_addons as tfa
from spektral.layers.convolutional.conv import Conv


class ImportanceLayer(Conv):
    def __init__(self, activation="reduce_weights", **kwargs):
        self.activation = activation
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        x, a = inputs
        # print(x.shape, a.shape)

        if mask is not None:
            x *= mask[0]
        x = tf.squeeze(x, axis=-1)

        # print(x.shape)
        if self.activation == "sparsemax":
            output = tfa.activations.sparsemax(x)
        elif self.activation == "softmax":
            output = tf.nn.softmax(x)
        else:
            nodes_ind = x
            # nodes_ind = 1 / x
            # nodes_ind = tf.where(
            #     tf.math.is_inf(nodes_ind), tf.zeros_like(nodes_ind), nodes_ind
            # )
            output = nodes_ind / tf.reduce_sum(nodes_ind)
        output = tf.expand_dims(output, axis=-1)

        return output

    def config(self):
        return {"importance": 1}
