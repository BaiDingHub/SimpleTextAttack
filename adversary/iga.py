import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
import copy

from adversary.attack import Adversary


class IGAAdversary(Adversary):
    """  IGA attack method.  """

    def __init__(self, synonym_selector, target_model, iterations_num=20, pop_max_size=60, crossover_coeff=0.5, variation_coeff=0.01, max_perturbed_percent=0.25):
        super(IGAAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        self.iterations_num = iterations_num
        self.pop_max_size = pop_max_size
        self.crossover_coeff = crossover_coeff
        self.variation_coeff = variation_coeff

    def find_best_replace(self, clean_tokens, position, synonyms, ori_label, N=2):
        result_sentences = []
        score_list = []
        sentence_list = []
        for word in synonyms:
            clean_tokens_new = list(clean_tokens)
            clean_tokens_new[position] = word
            new_sentence = ' '.join(clean_tokens_new)
            sentence_list.append(new_sentence)
        if(len(synonyms) < 2):
            result_sentences = sentence_list
        else:
            logits_new = self.target_model.query(sentence_list, [ori_label]*len(sentence_list))[0]
            score_list = list(1 - self._softmax(logits_new)[:, ori_label])
            best_score_list = []
            for i in range(N):
                best_score_list.append(score_list.index(max(score_list)))
                score_list[score_list.index(max(score_list))] = float('-inf')
            for j in best_score_list:
                result_sentences.append(sentence_list[j])
        return result_sentences

    def generate_seed_pop(self, tokens, ori_label):
        seed_pop = []
        synset = []
        for i in range(len(tokens)):
            sear_word = tokens[i]
            synonyms = self.synonym_selector.find_synonyms(sear_word)
            synset.append(synonyms)
            result_sentences = self.find_best_replace(tokens, i, synonyms, ori_label)
            seed_pop.extend(result_sentences)
        return seed_pop, synset

    def judge_adv(self, pop, ori_label):
        success = False
        adv_label = ori_label
        if len(pop) == 0:
            return success, None, None
        logits, classifications = self.target_model.query(pop, [ori_label]*len(pop))
        for j, c in enumerate(classifications):
            if c != ori_label:
                success = True
                adv_label = c
                return success, pop[j], adv_label
        lowest_idx = np.argmin(self._softmax(logits)[:, ori_label], axis=-1)
        return success, pop[lowest_idx], adv_label

    def fitness_function(self, pop, sentence, ori_label, a=0.5):
        logits, classifications = self.target_model.query(pop, [ori_label]*len(pop))
        # The smaller of the original score of the output, the higher of the score we get.
        fitness_score_1 = 1 - self._softmax(logits)[:, ori_label]
        text_len = len(sentence.split())
        fitness_score_2 = np.array(
            [(self.check_diff(new_sentence, sentence) / text_len) for new_sentence in pop]
        )
        fitness_score = a * fitness_score_1 + (1 - a) * fitness_score_2
        return fitness_score

    def select_high_fitness(self, sentence, pop, ori_label):
        if len(pop) <= self.pop_max_size:
            return pop
        all_score_list = list(self.fitness_function(pop, sentence, ori_label))
        best_allscore_list = []
        for i in range(self.pop_max_size):
            best_allscore_list.append(all_score_list.index(max(all_score_list)))
            all_score_list[all_score_list.index(max(all_score_list))] = float("inf")
        new_pop = []
        for score_index in best_allscore_list:
            new_pop.append(pop[score_index])
        return new_pop

    def crossover(self, pop):
        if len(pop) <= 2:
            return pop
        for i in range(len(pop)):
            temp = pop[i]
            pop[i] = temp.split()
        new_pop = pop.copy()
        for i in range(len(pop)):
            if np.random.randn() < self.crossover_coeff:
                j = random.randint(1, len(pop) - 1)
                k = random.randint(0, len(pop[i]) - 1)
                new_pop[i] = pop[i][0:k] + pop[j][k:len(pop[j])]
        for i in range(len(new_pop)):
            new_pop[i] = ' '.join(new_pop[i])
        return new_pop

    def variation(self, pop, ori_label, synset):
        for i in range(len(pop)):
            temp = pop[i]
            pop[i] = temp.split()
        new_pop = []
        for sentence in pop:
            if np.random.randn() < self.variation_coeff:
                j = random.randint(0, len(sentence) - 1)
                synonyms = synset[j]
                if len(synonyms) != 0:
                    result_sentences = self.find_best_replace(
                        sentence, j, synonyms, ori_label
                    )
                    new_pop.extend(result_sentences)
                else:
                    sentence = ' '.join(sentence)
                    new_pop.append(sentence)
            else:
                sentence = ' '.join(sentence)
                new_pop.append(sentence)
        return new_pop

    def run(self, sentence, ori_label): 
        clean_tokens = sentence.split()
        adv_sentence = sentence
        adv_label = ori_label
        success = False
        perturbed_tokens = list(clean_tokens)

        pop, synset = self.generate_seed_pop(perturbed_tokens, ori_label)
        for i in range(self.iterations_num):
            has_change_label, adv_sentence_tmp, adv_label_tmp = self.judge_adv(pop, ori_label)
            if adv_sentence_tmp:
                if (
                    self.check_diff(adv_sentence_tmp, sentence) / len(clean_tokens)
                    > self.max_perturbed_percent
                ):
                    break
                elif has_change_label:
                    adv_sentence = adv_sentence_tmp
                    adv_label = adv_label_tmp
                    success = True
                    break
                else:
                    adv_sentence = adv_sentence_tmp

            pop = self.select_high_fitness(sentence, pop, ori_label)
            pop = self.crossover(pop)
            pop = self.variation(pop, ori_label, synset)

        return success, adv_sentence, adv_label