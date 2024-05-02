import numpy as np
import tensorflow as tf
from preprocessing import get_data, ALL_CHARS, CHAR_MAP
import time

class CustomPadConv2D(tf.keras.layers.Layer):
    ''''
    Custom layer to pad the input tensor before applying a Conv2D layer.
    '''
    def __init__(self, filters, kernel_size, padding, **kwargs):
        super(CustomPadConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.padding = padding
        self.conv2d = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = "valid", strides = 1)
    
    def call(self, x):
        # Custom padding is (# images, height, width, channels)
        x_padded = tf.pad(tensor = x, paddings= [[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]])
        return self.conv2d(x_padded)

class small_basic_block(tf.keras.layers.Layer):
    '''
    Small basic block as defined in the paper. It is a sequence of 4 Conv2D layers with ReLU activations.
    '''
    def __init__(self, channel_in, channel_out):
        super(small_basic_block, self).__init__()
        self.basic_block = tf.keras.Sequential(
            layers = [
                tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = 1, activation="relu"),
                # padding is more basic in tf, so this is not an exact replica of the padding used in the paper
                # tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = (3,1), padding="same", activation="relu"),
                # tf.keras.layers.Conv2D(filters = (channel_out // 4), kernel_size = (1,3), padding="same", activation="relu"),
                CustomPadConv2D(filters = (channel_out // 4), kernel_size = (3,1), padding=(1,0)),
                CustomPadConv2D(filters = (channel_out // 4), kernel_size = (1,3), padding=(0,1)),
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
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1)),
                small_basic_block(64, 128),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 6
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,2)),
                small_basic_block(64,256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 10
                small_basic_block(256,256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 13
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,2)),
                tf.keras.layers.Dropout(rate=dropout_rate),
                tf.keras.layers.Conv2D(filters = 256, kernel_size = (1,4)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 18
                tf.keras.layers.Dropout(rate=dropout_rate),
                tf.keras.layers.Conv2D(filters = class_num, kernel_size = (13,1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(), # 22
            ]
        )

        self.container = tf.keras.layers.Conv2D(filters=class_num, kernel_size=(1, 1), strides=(1, 1))

        self.optimizer = tf.keras.optimizers.legacy.Adam()

    def call(self, image): # Image shape: (2 x 24 x 94 x 3)
        keep_features = list()
        for i,layer in enumerate(self.model.layers):
            image = layer(image)
            # print(image.shape)
            if i in [2, 6, 13, 22]: # ReLU layers
                keep_features.append(image)
                # print("image shape in keep features", image.shape)
        # print("keep features length", len(keep_features))
        
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
            f_mean = tf.math.reduce_mean(f_pow) # TODO is this right?
            f = f / f_mean
            global_context.append(f)

        
        x = tf.concat(values=global_context, axis=3)
        # x = tf.keras.layers.Conv2D(filters=class_num, kernel_size=(1, 1), strides=(1, 1))
        x = self.container(x)
        logits = tf.math.reduce_mean(x, axis=1) # Over time
        # should reduce across time (axis should be time)
        # print("logits:", logits)
        print("logits shape;", logits.shape)

        return logits
    

    def loss(self, logits, labels):
        """
        Calculates CTC loss. We are using CTC loss based on the paper we're using as reference.
        as it doesn't take one-hot encoded vectors and instead taks a string representation of
        the license plate to calculate losses.
        """

        # Example label sequences 
        # labels_sequences = [
        #     [1, 2, 3, 4],      # First sequence in the batch
        #     [5, 6]             # Second sequence in the batch
        # ]

        # Building the SparseTensor for CTC loss
        indices = []  
        values = [] 
        for batch_index, seq in enumerate(labels):
            for seq_index, label in enumerate(seq):
                indices.append([batch_index, seq_index])
                values.append(label)
        sparse_labels = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[2, 7]  # Shape [batch_size, max_sequence_length] # TODO: CHANGE TO 5000
        )

        # Transpose logits to T x N x C format
        # T represents the number of time steps in the sequence
        # N represents the batch size
        # C represents the number of classes
        logits = tf.transpose(logits, perm=(1, 0, 2))
        logits = tf.nn.log_softmax(logits, axis=2)

        print("logits shape:", logits.shape)
        print("labels shape:", labels.shape)

        # List of 7s with length 5000 TODO: Change back to 5000
        len_logits = tf.fill([2], 18)
        len_labels = tf.fill([2], 7)
        blank_index = 36 # last index in classes

        # Normalize logit values
        scaled_logits = logits / tf.reduce_max(tf.abs(logits), axis=-1, keepdims=True)
        log_probs = tf.nn.log_softmax(scaled_logits) # TODO: MAYBE DON'T NEED

        loss = tf.nn.ctc_loss(sparse_labels, log_probs, len_labels, len_logits, blank_index=blank_index, logits_time_major=True)
        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy
        """
        # TODO ask about this
        epoch_size = len(datasets) // args.test_batch_size
        batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        t1 = time.time()
        for i in range(epoch_size):
            # load train data
            images, labels, lengths = next(batch_iterator)
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start+length]
                targets.append(label)
                start += length
            targets = np.array([el.numpy() for el in targets])

            if args.cuda:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # forward
            prebs = Net(images)
            # greedy decode
            prebs = prebs.cpu().detach().numpy()
            preb_labels = list()
            for i in range(prebs.shape[0]):
                preb = prebs[i, :, :]
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0))
                no_repeat_blank_label = list()
                pre_c = preb_label[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label: # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)
            for i, label in enumerate(preb_labels):
                if len(label) != len(targets[i]):
                    Tn_1 += 1
                    continue
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    Tp += 1
                else:
                    Tn_2 += 1

        Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
        print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
        t2 = time.time()
        print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))
            
        return 0

def train(model, train_inputs, train_labels):
    acc = []
    input_shape = (None, 24, 94, 3)
    model.build(input_shape) # Input shape

    # dummy_input = tf.random.normal([1, input_features]) 
    # _ = model(dummy_input) 
    print(model.summary())

    with tf.GradientTape() as tape:
        # forward pass
        logits = model(train_inputs)
        loss = model.loss(logits, train_labels)
        print("OMG REAL LOSS", loss) 
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # acc.append(model.accuracy(logits, train_labels))

data = get_data()
train(LPRNetModel(), data[0][0:2], data[1][0:2])