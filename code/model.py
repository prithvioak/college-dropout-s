import numpy as np
import tensorflow as tf


class small_basic_block(tf.keras.Module):
    def __init__(self, channel_in, channel_out):
        pass
    def forward(self, x):
        pass


class LPRNetModel(tf.keras.Model):

    def __init__(self, lpr_max_len, phase, class_num, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, images):
        pass