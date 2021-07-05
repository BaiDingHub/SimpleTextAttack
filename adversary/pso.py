import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
import copy

from adversary.attack import Adversary



class PSOAdversary(Adversary):
    """ Particle Swarm Optimization Attack. """

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25, pop_max_size=60, iterations_num=40, max_seq_length=500):
        
        super(PSOAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        self.max_iters = iterations_num
        self.pop_size = pop_max_size
        self.max_seq_length = max_seq_length

    def do_replace(self, x_cur, pos, new_word): # Done
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def predict_batch(self, sentences): # Done
        seqs = [" ".join(words) for words in sentences]
        tem, _ = self.target_model.query(seqs, None)
        tem = self._softmax(tem)
        return tem

    def predict(self, sentence): # Done
        tem, _ = self.target_model.query([" ".join(sentence)], None)
        tem = self._softmax(tem[0])
        return tem

    def select_best_replacement(self, pos, x_cur, x_orig, ori_label, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """

        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w else x_cur for w in replace_list]
        new_x_preds = self.predict_batch(new_x_list)

        x_scores = 1 - new_x_preds[:, ori_label]
        orig_score = 1 - self.predict(x_cur)[ori_label]

        new_x_scores = x_scores - orig_score

        if (np.max(new_x_scores) > 0):
            best_id = np.argsort(new_x_scores)[-1]
            if np.argmax(new_x_preds[best_id]) != ori_label:
                return [1, new_x_list[best_id]]
            return [x_scores[best_id], new_x_list[best_id]]
        return [orig_score, x_cur]

    def perturb(self, x_cur, x_orig, neigbhours, w_select_probs, ori_label):
        # Pick a word that is not modified and is not UNK
        
        x_len = w_select_probs.shape[0]
 
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        replace_list = neigbhours[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, ori_label, replace_list)

    def generate_population(self, x_orig, neigbhours_list, w_select_probs, ori_label, pop_size):
        pop = []
        pop_scores=[]
        for i in range(pop_size):
            tem = self.perturb(x_orig, x_orig, neigbhours_list, w_select_probs, ori_label)
            if tem is None:
                return None
            if tem[0] == 1:
                return [tem[1]]
            else:
                pop_scores.append(tem[0])
                pop.append(tem[1])
        return pop_scores, pop

    def turn(self, x1, x2, prob, x_len):
        x_new = copy.deepcopy(x2)
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                x_new[i] = x1[i]
        return x_new

    def equal(self, a, b):
        if a == b:
            return -3
        else:
            return 3

    def sigmod(self, n):
        return 1 / (1 + np.exp(-n))

    def count_change_ratio(self, x, x_orig, x_len):
        change_ratio = float(np.sum(np.array(x) != np.array(x_orig))) / float(x_len)
        return change_ratio

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

        if np.sum(neighbours_len) == 0:
            return False, sentence, ori_label

        print(neighbours_len)

        tem = self.generate_population(x_orig, neigbhours_list, w_select_probs, ori_label, self.pop_size)
        if tem is None:
            return False, sentence, ori_label
        if len(tem) == 1:
            return self.check_return(tem[0], x_orig, ori_label)
        pop_scores, pop = tem
        part_elites = copy.deepcopy(pop)
        part_elites_scores = pop_scores
        all_elite_score = np.max(pop_scores)
        pop_ranks = np.argsort(pop_scores)
        top_attack = pop_ranks[-1]
        all_elite = pop[top_attack]

        Omega_1 = 0.8
        Omega_2 = 0.2
        C1_origin = 0.8
        C2_origin = 0.2
        V = [np.random.uniform(-3, 3) for rrr in range(self.pop_size)]
        V_P = [[V[t] for rrr in range(x_len)] for t in range(self.pop_size)]

        for i in range(self.max_iters):

            Omega = (Omega_1 - Omega_2) * (self.max_iters - i) / self.max_iters + Omega_2
            C1 = C1_origin - i / self.max_iters * (C1_origin - C2_origin)
            C2 = C2_origin + i / self.max_iters * (C1_origin - C2_origin)

            for id in range(self.pop_size):

                for dim in range(x_len):
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                                self.equal(pop[id][dim], part_elites[id][dim]) + self.equal(pop[id][dim],
                                                                                            all_elite[dim]))
                turn_prob = [self.sigmod(V_P[id][d]) for d in range(x_len)]
                P1 = C1
                P2 = C2
                # P1=self.sigmod(P1)
                # P2=self.sigmod(P2)

                if np.random.uniform() < P1:
                    pop[id] = self.turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self.turn(all_elite, pop[id], turn_prob, x_len)

            # pop_scores = []
            # pop_scores_all=[]
            # for a in pop:
            #     pt = self.predict(a)
            #     pop_scores.append(1 - pt[ori_label])
            #     pop_scores_all.append(pt)

            pop_scores_all = self.predict_batch(pop)
            pop_scores = 1 - pop_scores_all[:, ori_label]

            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]

            print('\t\t', i, ' -- ', pop_scores[top_attack])
            for pt_id in range(len(pop_scores_all)):
                pt = pop_scores_all[pt_id]
                if np.argmax(pt) != ori_label:
                    return self.check_return(pop[pt_id], x_orig, ori_label)

            new_pop = []
            new_pop_scores=[]
            for id in range(len(pop)):
                x=pop[id]
                change_ratio = self.count_change_ratio(x, x_orig, x_len)
                p_change = 1 - 2*change_ratio
                if np.random.uniform() < p_change:
                    tem = self.perturb(x, x_orig, neigbhours_list, w_select_probs, ori_label)
                    if tem is None:
                        return False, sentence, ori_label
                    if tem[0] == 1:
                        return self.check_return(tem[1], x_orig, ori_label)
                    else:
                        new_pop_scores.append(tem[0])
                        new_pop.append(tem[1])
                else:
                    new_pop_scores.append(pop_scores[id])
                    new_pop.append(x)
            pop = new_pop

            pop_scores = new_pop_scores
            pop_ranks = np.argsort(pop_scores)
            top_attack = pop_ranks[-1]
            for k in range(self.pop_size):
                if pop_scores[k] > part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            elite = pop[top_attack]
            if np.max(pop_scores) > all_elite_score:
                all_elite = elite
                all_elite_score = np.max(pop_scores)
        
        return False, sentence, ori_label