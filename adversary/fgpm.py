import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
import copy

from adversary.attack import Adversary


class FGPMAdversary(Adversary):
    """  FGPM attack method.  """

    def __init__(self, synonym_selector, target_model, max_iter=20, max_perturbed_percent=0.25):
        super(FGPMAdversary, self).__init__(synonym_selector, target_model, max_perturbed_percent)
        self.max_iter = max_iter

    def _find_synonym(self, xs, dist_mat, threshold=0.5):
        synonyms = dist_mat[:, :, 0][xs]
        synonyms_dist = dist_mat[:, :, 1][xs]
        synonyms = torch.where(synonyms_dist <= threshold, synonyms, torch.zeros_like(synonyms))
        synonyms = torch.where(synonyms >= 0, synonyms, torch.zeros_like(synonyms)) # [None, Sequence_len, Syn_num]
        return synonyms 

    def run_batch(self, xs, ys, xs_mask, dist_mat):
        """ args: xs: Tensor, a batch of encoded input samples. """
        
        adv_xs = xs  # b, n, d

        # real words num in each sample, used to calculate whether the substitution rate is exceeded
        words_num = torch.sum(xs_mask, dim=-1) 
        batch_size = adv_xs.shape[0]
        synonyms = self._find_synonym(xs, dist_mat).long()
        modified_mask = torch.zeros_like(xs_mask)
        loss_fn = torch.nn.CrossEntropyLoss()

        for i in range(self.max_iter):
            embeddings = self.target_model.get_embeddings()
            embedded_chars = self.target_model.input_to_embs(adv_xs)
            embedded_chars = embedded_chars.clone().detach().to(torch.float).requires_grad_(True)

            logits = self.target_model.embs_to_logit(embedded_chars)
            predictions = torch.argmax(logits, dim=-1)
            loss = loss_fn(logits, ys)

            modified_num = torch.sum(modified_mask, dim=-1)
            modified_ratio = torch.div(modified_num + 1.0, words_num)

            # whether the sample becomes an adversarial example
            unsuc_mask = torch.eq(predictions, ys) & \
                        torch.le(modified_ratio, self.max_perturbed_percent)

            if torch.sum(unsuc_mask).detach().cpu() == 0:
                break

            # step 1: get Jacobian matrix
            loss.backward()
            # p_direction = torch.autograd.grad(loss, embedded_chars)[0]
            p_direction = embedded_chars.grad

            # step 2: compute projection
            synonyms_embed = embeddings[synonyms]
            xs_embed = torch.unsqueeze(embedded_chars, -2)
            p_direction = torch.unsqueeze(p_direction, -2)
            projection = torch.sum(
                torch.mul(synonyms_embed - xs_embed, p_direction), dim=-1)

            # step 3: mask projection
            synonym_mask = torch.le(synonyms, 0).float()
            projection = projection - 1000000.0 * synonym_mask

            # step 4: substitution
            value, pos = torch.max(torch.max(projection, dim=-1).values, dim=-1)
            # note that those samples which have become adversarial examples are not considered,
            # and thus the subscript [unsuc_mask] is added
            adv_xs[torch.arange(batch_size)[unsuc_mask], pos[unsuc_mask]] = \
                synonyms[torch.arange(batch_size)[unsuc_mask], pos[unsuc_mask], torch.argmax(projection[torch.arange(batch_size)[unsuc_mask], pos[unsuc_mask]], dim=-1)]

            # step 5: update global state
            modified_mask[torch.arange(batch_size)[unsuc_mask], pos[unsuc_mask]] = 1
            # the synonym used cannot be used again, so updated to 0
            synonyms[torch.arange(batch_size)[unsuc_mask], pos[unsuc_mask], torch.argmax(projection[torch.arange(batch_size)[unsuc_mask], pos[unsuc_mask]], dim=-1)] = 0

        logits = self.target_model.input_to_logit(adv_xs)
        predictions = torch.argmax(logits, dim=-1)
        suc_index = torch.ne(predictions, ys)

        return adv_xs, suc_index, predictions, modified_mask

    def run(self):
        raise NotImplementedError