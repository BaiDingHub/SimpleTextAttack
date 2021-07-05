import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
import copy

from adversary.attack import Adversary

class GAAdversary(Adversary):
    """  GA attack method.  """

    def __init__(self, synonym_selector, target_model, iterations_num=20, pop_max_size=60, max_perturbed_percent=0.25):
        super(GAAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        self.max_iters = iterations_num
        self.pop_size = pop_max_size
        self.temp = 0.3

    def predict_batch(self, sentences): # Done
        seqs = [" ".join(words) for words in sentences]
        tem, _ = self.target_model.query(seqs, None)
        tem = self._softmax(tem)
        return tem

    def predict(self, sentence): # Done
        tem, _ = self.target_model.query([" ".join(sentence)], None)
        tem = self._softmax(tem[0])
        return tem

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, pos, x_cur, x_orig, ori_label, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w else x_cur for w in replace_list]
        new_x_preds = self.predict_batch(new_x_list)

        new_x_scores = 1 - new_x_preds[:, ori_label]
        orig_score = 1 - self.predict(x_cur)[ori_label]
        new_x_scores = new_x_scores - orig_score

        if (np.max(new_x_scores) > 0):
            return new_x_list[np.argsort(new_x_scores)[-1]]
        return x_cur
    
    def perturb(self, x_cur, x_orig, neigbhours, w_select_probs, ori_label):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(np.array(x_orig) != np.array(x_cur)) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        replace_list = neigbhours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, ori_label, replace_list)

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, ori_label, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, ori_label) for _ in range(pop_size)]

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def check_return(self, perturbed_words, ori_words, ori_label):
        perturbed_text = " ".join(perturbed_words.tolist())
        clean_text = " ".join(ori_words.tolist())
        if self.check_diff(clean_text, perturbed_text) / len(ori_words) > self.max_perturbed_percent:
            return False, clean_text, ori_label
        else:
            adv_label = self.target_model.query([perturbed_text], [ori_label])[1][0]
            assert (adv_label != ori_label)
            return True, perturbed_text, adv_label

    def run(self, sentence, ori_label):

        x_orig = np.array(sentence.split())
        x_len = len(x_orig)

        neigbhours_list = []
        for i in range(x_len):
            neigbhours_list.append(self.synonym_selector.find_synonyms(x_orig[i]))
            
        neighbours_len = [len(x) for x in neigbhours_list]
        w_select_probs = []
        for pos in range(x_len):
            if neighbours_len[pos] == 0:
                w_select_probs.append(0)
            else:
                w_select_probs.append(min(neighbours_len[pos], 10))
        w_select_probs = w_select_probs / np.sum(w_select_probs)

        pop = self.generate_population(
            x_orig, neigbhours_list, w_select_probs, ori_label, self.pop_size)
        for i in range(self.max_iters):
            pop_preds = self.predict_batch(pop)
            pop_scores = 1 - pop_preds[:, ori_label]
            print('\t\t', i, ' -- ', np.max(pop_scores))
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            logits = np.exp(pop_scores / self.temp)
            select_probs = logits / np.sum(logits)

            if np.argmax(pop_preds[top_attack, :]) != ori_label:
                return self.check_return(pop[top_attack], x_orig, ori_label)
            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)

            childs = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size-1)]
            childs = [self.perturb(
                x, x_orig, neigbhours_list, w_select_probs, ori_label) for x in childs]
            pop = elite + childs

        return False, sentence, ori_label