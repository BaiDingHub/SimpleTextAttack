import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
import copy

from adversary.attack import Adversary

class GreedyAdversary(Adversary):
    """  GSA attack method.  """

    def __init__(self, synonym_selector, target_model, task_name, emb_mat, vocab, max_perturbed_percent=0.25):
        super(GreedyAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        if task_name == 'imdb':
            self.thought_thres = 0.2
        elif task_name == 'agnews':
            self.thought_thres = 0.45
        else:
            self.thought_thres = 0.3
        self.task_name = task_name
        self.emb_mat = emb_mat
        self.vocab = vocab

    def semantic_similarity(self, thought_vec_a, thought_vec_b):
        semantic_sim = np.sqrt(np.sum(np.square(thought_vec_a - thought_vec_b)))
        return semantic_sim

    def replace_word(self, x, position, word):
        x = x.split()
        x_new = x
        x_new[position] = word
        x_new = ' '.join(x_new)
        return x_new

    def replace(self, sentence, position, near, org_thought_vec):
        result_sentences = []
        x_split = sentence.split()
        for word in near:
            if word == x_split[position]:
                continue
            new_sentence = self.replace_word(sentence, position, word)
            new_thought_vector = self.get_thought_vector(new_sentence)
            semantic_sim = self.semantic_similarity(org_thought_vec, new_thought_vector)
            if semantic_sim < self.thought_thres:
                result_sentences.append(new_sentence)
        return result_sentences


    def get_thought_vector(self, x):
        x_split = x.split()
        vectors = []
        for w in x_split:
            if w in self.vocab:
                encoded_w = self.vocab[w]
                if np.sum(np.array(self.emb_mat[:, encoded_w])) != 0:
                    vectors.append(self.emb_mat[:, encoded_w])
        return np.mean(vectors, axis=0)

    def run(self, sentence, ori_label): 
        clean_tokens = sentence.split()
        adv_sentence = sentence
        adv_label = ori_label
        success = False
        perturbed_tokens = list(clean_tokens)

        ori_thought_vec = self.get_thought_vector(sentence)
        synonyms_dict = {}
        for w in perturbed_tokens:
            synonyms_dict[w] = self.synonym_selector.find_synonyms(w)

        loop_num = 0
        while self.check_diff(adv_sentence, sentence) + 1 <= self.max_perturbed_percent * len(
            perturbed_tokens
        ) and loop_num < len(perturbed_tokens):
            loop_num += 1
            result_sentences = []
            for i, w in enumerate(perturbed_tokens):
                if w in synonyms_dict and perturbed_tokens[i] == clean_tokens[i]:
                    near = synonyms_dict[w]
                    result_sentences += self.replace(adv_sentence, i, near, ori_thought_vec)
            if len(result_sentences) >= 1:
                logits, classifications = self.target_model.query(result_sentences, [ori_label]*len(result_sentences))
            else:
                break
            scores = self._softmax(logits)[:, ori_label]
            idx = np.argmin(scores)
            if classifications[idx] != ori_label:
                success = True
                adv_label = classifications[idx]
                return success, result_sentences[idx], adv_label
            else:
                adv_sentence = result_sentences[idx]
                perturbed_tokens = adv_sentence.split()

        return success, adv_sentence, adv_label