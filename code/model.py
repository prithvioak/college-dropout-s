import numpy as np
import tensorflow as tf
from preprocessing import get_data

class CustomPadConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, **kwargs):
        super(CustomPadConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.padding = padding
        self.conv2d = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = "valid", strides = 1)
    
    def call(self, x):
        # Custom padding, TODO: check if this is the correct way to pad
        x_padded = tf.pad(x, [[0, 0], [self.padding[0], self.padding[0]], [0, 0], [0, 0]])
        return self.conv2d(x_padded)

class small_basic_block(tf.keras.layers.Layer):
    def __init__(self, channel_in, channel_out):
        super(small_basic_block, self).__init__()
        self.basic_block = tf.keras.Sequential(
            layers = [
                tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = 1, activation="relu"),
                # padding is more basic in tf, so this is not an exact replica of the padding used in the paper
                tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = (3,1), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = (1,3), padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters = channel_out, kernel_size=1)
            ])
    def call(self, x):
        return self.basic_block(x)


class LPRNetModel(tf.keras.Model):

    def __init__(self, lpr_max_len=0, phase=0, class_num=37, dropout_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.keras.Sequential(
            layers = [
                tf.keras.layers.Conv2D(filters = 64, kernel_size = 3),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 2 
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding="same"),
                small_basic_block(64, 128),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 6
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,2), padding="same"),
                small_basic_block(64,256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                small_basic_block(256,256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,2), padding="same"),
                tf.keras.layers.Dropout(rate=dropout_rate),
                tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,4)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 11
                tf.keras.layers.Dropout(rate=dropout_rate),
                tf.keras.layers.Conv2D(filters = class_num, kernel_size = (1,13)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 15
            ]
        )

        self.optimizer = tf.keras.optimizers.legacy.Adam()

    def call(self, image):
        keep_features = list()
        for i,layer in enumerate(self.model.layers):
            image = layer(image)
            if i in [2, 6, 11, 15]: # ReLU layers
                keep_features.append(image)
        global_context = list()
        # build global_context
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                # f = tf.nn.AvgPool2d(kernel_size=5, stride=5)(f)
                f = tf.keras.layers.AveragePooling2D(pool_size=5, strides=5)(f)
            if i in [2]:
                # f = tf.nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
                f = tf.keras.layers.AveragePooling2D(pool_size=(4, 10), strides=(4, 2))(f)
            f_pow = tf.math.square(f)
            f_mean = tf.math.reduce_mean(f_pow)
            f = f / f_mean
            global_context.append(f)
        
        x = tf.concat(values=global_context, axis=-1)
        x = self.container(x)
        logits = tf.math.reduce_mean(x, axis=2)
        print("logits:", logits)

        return logits
    

    def loss(self, logits, labels, len_logits, len_labels):
        """
        Calculates CTC loss. We are using CTC loss based on the paper we're using as reference.
        as it doesn't take one-hot encoded vectors and instead taks a string representation of
        the license plate to calculate losses.
        """
        loss = tf.nn.ctc_loss(labels, logits, len_labels, len_logits)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy
        """
        # TODO

def train(model, train_inputs, train_labels):
    with tf.GradientTape() as tape:
        # forward pass
        logits = model.call(train_inputs)
        loss = model.loss(logits, train_labels)
        print("loss: ", loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

data = get_data()
train(LPRNetModel(), data[0][0:2], data[1][0:2])