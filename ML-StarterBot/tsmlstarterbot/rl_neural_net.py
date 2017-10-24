import os
import tensorflow as tf
import numpy as np

# We don't want tensorflow to produce any warnings in the standard output, since the bot communicates
# with the game engine through stdout/stdin.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


# ReinforcementLearningNN: LOTS TO BE DONE
class RLNeuralNet(object):
    activations = {}
    parameters = {}

    def __init__(self, cached_model=None, seed=None, architecture=[64, 128, 256, 64], dropout=0.7):
        pass

    def train(self):
        pass

    def predict(self, input_data):
       pass

    def compute_loss(self, input_data, expected_output_data):
       pass

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)
