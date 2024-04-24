import numpy as np
import tensorflow as tf


class small_basic_block(tf.keras.Module):
    def __init__(self, channel_in, channel_out):
        self.basic_block = tf.keras.Sequential(
            layers = [
                tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = 1, activation="relu"),
                # padding is more basic in tf, so this is not an exact replica of the padding used in the paper
                tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = (3,1), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = (1,3), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters = channel_out, kernel_size=1)
            ])
    def forward(self, x):
        return self.basic_block(x)


class LPRNetModel(tf.keras.Model):

    def __init__(self, lpr_max_len, phase, class_num, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, images):
        pass