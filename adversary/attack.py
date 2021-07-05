import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
import copy

class Adversary(object):
    """An Adversary tries to fool a model on a given example."""

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25):
        self.synonym_selector = synonym_selector
        self.target_model = target_model
        self.max_perturbed_percent = max_perturbed_percent

    def run(self, model, dataset, device, opts=None):
        """Run adversary on a dataset.
        Args:
        model: a TextClassificationModel.
        dataset: a TextClassificationDataset.
        device: torch device.
        Returns: pair of
        - list of 0-1 adversarial loss of same length as |dataset|
        - list of list of adversarial examples (each is just a text string)
        """
        raise NotImplementedError

    def _softmax(self, x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            _c_matrix = np.max(x, axis=1)
            _c_matrix = np.reshape(_c_matrix, [_c_matrix.shape[0], 1])
            _diff = np.exp(x - _c_matrix)
            x = _diff / np.reshape(np.sum(_diff, axis=1), [_c_matrix.shape[0], 1])
        else:
            _c = np.max(x)
            _diff = np.exp(x - _c)
            x = _diff / np.sum(_diff)
        assert x.shape == orig_shape
        return x

    def check_diff(self, sentence, perturbed_sentence):
        words = sentence.split()
        perturbed_words = perturbed_sentence.split()
        diff_count = 0
        if len(words) != len(perturbed_words):
            raise RuntimeError("Length changed after attack.")
        for i in range(len(words)):
            if words[i] != perturbed_words[i]:
                diff_count += 1
        return diff_count