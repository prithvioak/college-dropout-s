import numpy as np
import tensorflow as tf
from preprocessing import get_data_segmented, ALL_CHARS, CHAR_MAP
import time
import matplotlib.pyplot as plt



class SegmentationModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 50
        self.num_classes = 36
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)

        self.model = tf.keras.Sequential([
            # TODO: Add layers
            tf.keras.layers.Conv2D(filters = 64, kernel_size = 3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = 3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = 3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            
        ])
        # self.output_layer1 = tf.keras.layers.Dense(self.num_classes, activation = 'relu')
        self.output_layer2 = tf.keras.layers.Dense(self.num_classes) # Softmax?


    def call(self, x): # Input shape: (batch_size, num_chars_per_image=7, height=32, width=24, channels=3)
        x = self.model(x)
        x = tf.reshape(x, (x.shape[0], 7, -1))
        # x = self.output_layer1(x)
        x = self.output_layer2(x)

        return x


    def loss(self, logits, labels):
        '''
        Softmax cross entropy loss
        '''
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return tf.reduce_mean(loss)

    def mean_accuracy(self, logits, labels):
        '''
        Compute accuracy of model predictions
        '''
        correct_predictions = tf.equal(tf.argmax(logits, 2), tf.argmax(labels, 2))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def tp_rate(self, logits, labels):
        '''
        Computes the true positive rate (all 7 characters are correct) 
        '''
        correct_predictions = tf.equal(tf.argmax(logits, axis=2), tf.argmax(labels, axis=2))
        correct_plates = tf.reduce_all(correct_predictions, axis=1) # boolean tensor of shape (batch_size,)
        return tf.reduce_mean(tf.cast(correct_plates, tf.float32)) # Return mean of true positives count
    
    def tp_count(self, logits, labels):
        '''
        Computes the number of true positive license plates
        '''
        correct_predictions = tf.equal(tf.argmax(logits, axis=2), tf.argmax(labels, axis=2))
        correct_plates = tf.reduce_all(correct_predictions, axis=1)
        return tf.reduce_sum(tf.cast(correct_plates, tf.int32)) # RETURNS NUMBER OF TRUE POSITIVES
    
    def per_char_accuracy(self, logits, labels):
        '''
        Computes the accuracy of each character in the license plate
        '''
        accuracies = []


def train(model, train_inputs, train_labels):

    # Shuffle inputs and labels and split into batches
    shuffled_indices = tf.random.shuffle(tf.range(tf.shape(train_inputs)[0])) #Shuffle indices of the input examples
    train_inputs = tf.gather(train_inputs, shuffled_indices)
    train_labels = tf.gather(train_labels, shuffled_indices)
    num_batches = len(train_inputs) // model.batch_size
    batch_inputs = np.array_split(train_inputs, num_batches)
    batch_labels = np.array_split(train_labels, num_batches)

    losses = []
    for epoch in range(10):
        epoch_loss = []

        for batch in range(num_batches):
            
            batch_i_inputs = batch_inputs[batch]
            batch_i_labels = batch_labels[batch]

            with tf.GradientTape() as tape:
                logits = model(batch_i_inputs)
                loss = model.loss(logits, batch_i_labels)
                epoch_loss.append(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        logits = model(train_inputs)
        print(f"Epoch {epoch} Train Character-Wise Accuracy: {model.mean_accuracy(logits, train_labels)}")
        print(f"Epoch {epoch} Train True Positive Rate: {model.tp_rate(logits, train_labels)}")
        print(f"Epoch {epoch} Train True Positive Count: {model.tp_count(logits, train_labels)}")

        epoch_loss = np.array(epoch_loss)
        epoch_loss = np.mean(epoch_loss)
        losses.append(epoch_loss)
    losses = np.array(losses)
    visualize_loss(losses)
        
    return None


def test(model, test_inputs, test_labels):
    print(f"Test Character-Wise Accuracy: {model.mean_accuracy(model(test_inputs), test_labels)}")
    print(f"Test True Positive Rate: {model.tp_rate(model(test_inputs), test_labels)}")
    print(f"Test True Positive Count: {model.tp_count(model(test_inputs), test_labels)}")
    return None

def visualize_loss(losses): 
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Segmentation - Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()  


data = get_data_segmented()
threshold = 500
model = SegmentationModel()
train(model, data[0][0:threshold], data[1][0:threshold])
test(model, data[0][threshold:threshold+200], data[1][threshold:threshold+200])
