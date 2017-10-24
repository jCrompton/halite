import os
import tensorflow as tf
import numpy as np

from tsmlstarterbot.common import PLANET_MAX_NUM, PER_PLANET_FEATURES

# We don't want tensorflow to produce any warnings in the standard output, since the bot communicates
# with the game engine through stdout/stdin.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


# Normalize planet features within each frame.
def normalize_input(input_data):

    # Assert the shape is what we expect
    shape = input_data.shape
    assert len(shape) == 3 and shape[1] == PLANET_MAX_NUM and shape[2] == PER_PLANET_FEATURES

    m = np.expand_dims(input_data.mean(axis=1), axis=1)
    s = np.expand_dims(input_data.std(axis=1), axis=1)
    return (input_data - m) / (s + 1e-6)


class NeuralNet(object):
    activations = {}
    parameters = {}

    def __init__(self, cached_model=None, seed=None, architecture=[12, 48, 12, 6], dropout=0.6):
        self._graph = tf.Graph()
        self.architecture = [PER_PLANET_FEATURES] + architecture
        self.dropout = dropout

        with self._graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._session = tf.Session()
            self._features = tf.placeholder(dtype=tf.float32, name="input_features",
                                            shape=(None, PLANET_MAX_NUM, PER_PLANET_FEATURES))

            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self._target_distribution = tf.placeholder(dtype=tf.float32, name="target_distribution",
                                                       shape=(None, PLANET_MAX_NUM))

            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            flattened_frames = tf.reshape(self._features, [-1, PER_PLANET_FEATURES])

            # Initialize the parameters from custom architecture
            for i in range(len(self.architecture) - 1):
                w_str = 'W{}'.format(i + 1)
                b_str = 'b{}'.format(i + 1)
                Wi = tf.get_variable(w_str, [self.architecture[i], self.architecture[i + 1]],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=1))
                bi = tf.get_variable(b_str, [self.architecture[i + 1]],
                                     initializer=tf.contrib.layers.xavier_initializer(seed=1))
                self.parameters[w_str] = Wi
                self.parameters[b_str] = bi

            b1 = self.parameters['b1']
            self.activations['Z1'] = tf.add(tf.matmul(flattened_frames, self.parameters['W1']), b1)

            # Initialize the tensors from variables
            for i in range(len(self.architecture) - 2):
                Wi = self.parameters['W{}'.format(i + 2)]
                bi = self.parameters['b{}'.format(i + 2)]
                Ai = tf.nn.relu(self.activations['Z{}'.format(i + 1)])
                Ai_dropout = tf.nn.dropout(Ai, self.dropout)
                print(Ai_dropout.shape, Wi.shape, bi.shape, self.architecture, i)
                self.activations['Z{}'.format(i + 2)] = tf.add(tf.matmul(Ai_dropout, Wi), bi)

            final_layer = tf.contrib.layers.fully_connected(self.activations['Z{}'.format(len(self.architecture)-1)],
                                                            1, activation_fn=None)

            # Group the planets back in frames.
            logits = tf.reshape(final_layer, [-1, PLANET_MAX_NUM])

            self._prediction_normalized = tf.nn.softmax(logits)

            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self._target_distribution))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
            self._saver = tf.train.Saver()

            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        loss, _ = self._session.run([self._loss, self._optimizer],
                                    feed_dict={self._features: normalize_input(input_data),
                                               self._target_distribution: expected_output_data})
        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        return self._session.run(self._prediction_normalized,
                                 feed_dict={self._features: normalize_input(np.array([input_data]))})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :param input_data: numpy array of shape (number of frames, PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :param expected_output_data: numpy array of shape (number of frames, PLANET_MAX_NUM)
        :return: training loss on the input data
        """
        return self._session.run(self._loss,
                                 feed_dict={self._features: normalize_input(input_data),
                                            self._target_distribution: expected_output_data})

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)

