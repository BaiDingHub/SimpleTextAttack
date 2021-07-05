import numpy as np

from adversary.attack import Adversary

class PWWSAdversary(Adversary):
    """  PWWS attack method.  """

    def __init__(self, synonym_selector, target_model, max_perturbed_percent=0.25):
        super(PWWSAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)

    def R_func(self, clean_tokens, idx, candidates, ori_label):
        max_diff = -100
        max_word = clean_tokens[idx]
        sentence = ' '.join(clean_tokens)
        logits = self.target_model.query([sentence], [ori_label])[0]
        score = self._softmax(logits)[0][ori_label]
        sentence_new_list = []
        for c in candidates:
            clean_tokens_new = list(clean_tokens)
            clean_tokens_new[idx] = c
            sentence_new = ' '.join(clean_tokens_new)
            sentence_new_list.append(sentence_new)
        if len(sentence_new_list) != 0:
            logits_new = self.target_model.query(sentence_new_list, [ori_label]*len(sentence_new_list))[0]
            score_new = self._softmax(logits_new)[:, ori_label]
            diff = score - score_new
            max_diff = np.max(diff)
            max_word = candidates[np.argmax(diff)]
        return max_word, max_diff


    def S_func(self, clean_tokens, ori_label):
        saliency_list = []
        sentence = ' '.join(clean_tokens)
        logits = self.target_model.query([sentence], [ori_label])[0]
        score = self._softmax(logits)[0][ori_label]
        sentence_new_list = []
        for i in range(len(clean_tokens)):
            clean_tokens_new = list(clean_tokens)
            clean_tokens_new[i] = '[UNK]'
            sentence_new = ' '.join(clean_tokens_new)
            sentence_new_list.append(sentence_new)
        logits_new = self.target_model.query(sentence_new_list, [ori_label]*len(sentence_new_list))[0]
        score_new = self._softmax(logits_new)[:, ori_label]
        saliency = score - score_new
        soft_saliency_list = list(self._softmax(saliency))
        return soft_saliency_list
    

    def H_func(self, clean_tokens, ori_label):
        saliency_list = self.S_func(clean_tokens, ori_label)
        result_list = []
        for i, w in enumerate(clean_tokens):
            candidates = self.synonym_selector.find_synonyms(w)
            max_word, max_diff = self.R_func(clean_tokens, i, candidates, ori_label)
            result_list.append([i, max_word, max_diff])
        score_list = [res[2] * saliency for res, saliency in zip(result_list, saliency_list)]
        indexes = np.argsort(np.array(score_list))[::-1]
        replace_list = []
        for index in indexes:
            res = result_list[index]
            replace_list.append([res[0], res[1]])
        return replace_list

    def run(self, sentence, ori_label): 
        clean_tokens = sentence.split()
        adv_sentence = sentence
        adv_label = ori_label
        success = False
        perturbed_tokens = list(clean_tokens)

        replace_list = self.H_func(clean_tokens, ori_label)
        for i in range(len(replace_list)):
            tmp = replace_list[i]
            perturbed_tokens[tmp[0]] = tmp[1]
            adv_sentence = ' '.join(perturbed_tokens)
            prediction = self.target_model.query([adv_sentence], [ori_label])[1][0]
            if int(prediction) != int(ori_label):
                success = True
                adv_label = prediction
                return success, adv_sentence, adv_label
            if self.check_diff(sentence, adv_sentence) + 1 > len(clean_tokens) * self.max_perturbed_percent:
                break
        return success, adv_sentence, adv_label