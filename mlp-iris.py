!/usr/bin/env python
"""
Example Tensorflow code to train a 2-layer (input, hidden, output) 
neural network machine learning model using the Iris data set. At the end,
the script computes the prediction accuracy for both the training and 
test sets.
"""

import tensorflow as tf
import numpy as np
import time

# Data sets for Iris. Neither file should have headers, so the training file 
# should have 120 rows, and the test file should have 30 rows. 
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Pass this configuration to tf.Session() to disable GPU.
CONFIG_CPU_ONLY = tf.ConfigProto(device_count = {'GPU' : 0})

class Config:
    learning_rate = 0.01            # Gradient descent learning rate.
    num_epochs = 10000              # Gradient descent number of iterations.
    H1_size = 10                    # Size of 1st (and only) hidden layer.
    regularization_strength = 0.1   # Regularization.

class Model:
    """
    Parameters for 2-layer NN model (input, hidden, output) learned
    during training.
    """
    W1 = None
    b1 = None
    W2 = None
    b2 = None

class Data:
    """
    Utility class for loading training and test CSV files.
    """
    def __init__(self):
        self.training_features = None
        self.training_labels = None
        self.training_labels_1hot = None
        self.test_features = None
        self.test_labels = None
        self.test_labels_1hot = None

    def load(self, training_filename, test_filename):
        """
        Load CSV files into class member variables.
        """

        # Load training data using load_csv() function from Tensorflow 0.10.
        training_set = tf.contrib.learn.datasets.base.load_csv(
            filename=training_filename, target_dtype=np.int, has_header=False)

        self.training_features    = training_set.data.astype(np.float32)
        self.training_labels      = training_set.target
        self.training_labels_1hot = self.convert_to_one_hot(self.training_labels) 

        # Load test data using load_csv() function from Tensorflow 0.10.
        test_set = tf.contrib.learn.datasets.base.load_csv(
            filename=test_filename, target_dtype=np.int, has_header=False)

        self.test_features    = test_set.data.astype(np.float32)
        self.test_labels      = test_set.target
        self.test_labels_1hot = self.convert_to_one_hot(self.test_labels) 

    def convert_to_one_hot(self, vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convert_to_one_hot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
        if num_classes is None:
            num_classes = np.max(vector)+1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)
        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)

class IrisClassifier:
    """
    Trains a 2-layer neural network model for classifying the Iris data set.
    """

    def __init__(self):
        self.data = None

    def loadData(self):
        """
        Load data from CSV files.
        """
        self.data = Data()
        self.data.load(IRIS_TRAINING, IRIS_TEST)

    def trainModel(self):
        """
        Trains a 2-layer NN model using TensorFlow.
        Layers: Input --> Hidden --> Output
        """
        num_features = self.data.training_features.shape[1]
        num_classes = self.data.training_labels_1hot.shape[1]

        # Create placeholders for the training data. Note that the 
        # number of rows is set to None so that different size data sets
        # (or batches) can be loaded.
        x_ph = tf.placeholder(tf.float32, [None, num_features])
        y_ph = tf.placeholder(tf.float32, [None, num_classes])

        # Construct hidden layer.
        W1 = tf.get_variable(name="W1", 
                             shape=[num_features, Config.H1_size], 
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1", 
                             shape=[Config.H1_size], 
                             initializer=tf.constant_initializer(0.0))
        H1 = tf.matmul(x_ph, W1) + b1
        H1 = tf.nn.relu(H1)

        # Construct output layer.
        W2 = tf.get_variable(name="W2", 
                             shape=[Config.H1_size, num_classes], 
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2", 
                             shape=[num_classes], 
                             initializer=tf.constant_initializer(0.0))
        y_hat = tf.matmul(H1, W2) + b2

        # Loss function. Computes cross-entropy loss between computed y_hat
        # and y_ph (which holds true values). The y_hat values are normalized
        # with softmax.
        J = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(y_hat, y_ph) + \
                Config.regularization_strength * tf.nn.l2_loss(W1) + \
                Config.regularization_strength * tf.nn.l2_loss(W2))

        train_step = tf.train.GradientDescentOptimizer(Config.learning_rate).minimize(J)

        sess = tf.Session(config=CONFIG_CPU_ONLY)
        sess.run(tf.initialize_all_variables())

        start_time = time.time()

        # --- Gradient descent loop. -----------------------------------------
        for i in range(Config.num_epochs):
            op, J_result = sess.run([train_step, J], feed_dict={x_ph:self.data.training_features, y_ph: self.data.training_labels_1hot})

            if (i % 1000 == 0):
                print "Epoch %6d/%6d: J=%10.5f" % (i, Config.num_epochs, J_result)
        # --------------------------------------------------------------------

        end_time = time.time()

        total_time_in_seconds = end_time-start_time
        print "Training took %.2f seconds" % total_time_in_seconds

        # Save the model parameters in case you need it.
        model = Model()
        model.W1, model.b1, model.W2, model.b2 = sess.run([W1, b1, W2, b2]) 

        # Compute accuracy on training set.
        correct_predictions_op = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_ph, 1)) # List of T,F
        accuracy_op = tf.reduce_mean(tf.cast(correct_predictions_op, tf.float32))
        correct_predictions, accuracy = \
            sess.run([correct_predictions_op, accuracy_op], 
                     feed_dict={x_ph:self.data.training_features, y_ph:self.data.training_labels_1hot})
        print
        print "Predictions on training data:"
        print correct_predictions
        print "Training accuracy = %.3f" % accuracy

        # Compute accuracy on test set.
        correct_predictions, accuracy = \
            sess.run([correct_predictions_op,i accuracy_op], 
                     feed_dict={x_ph:self.data.test_features, y_ph:self.data.test_labels_1hot})
        print
        print "Predictions on test data:"
        print correct_predictions
        print "Test accuracy = %.3f" % accuracy

        return model


def main():
    iris_classifier = IrisClassifier()

    # Load data from CSV files.
    iris_classifier.loadData()

    # Train the model.
    model = iris_classifier.trainModel()

    # Do something with 'model' if needed.


if __name__ == "__main__":
    main()
