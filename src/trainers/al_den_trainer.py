import logging
import math
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from data.data_loader import create_test_dataloader
from data.old_dataset import create_sequence_templates
from models.den import DEN
from trainers.base_trainer import BaseTrainer


class AlDenTrainer(BaseTrainer):

    def __init__(self, device, **kwargs) -> None:
        super().__init__(**kwargs)

        embedding_template, embedding_mask = create_sequence_templates()

        self.den = DEN(embedding_template=embedding_template, embedding_mask=embedding_mask, device=device)

    def compute_loss(self):
        pass
        #Define target isoform loss function
        # target_iso = np.zeros((len(target_isos), 1))
        # for i, t_iso in enumerate(target_isos) :
        #     target_iso[i, 0] = t_iso

        # masked_entropy_mse_region_1 = get_target_entropy_sme_masked(pwm_start=region_1_start, pwm_end=region_1_end, target_bits=region_1_target_bits)
        # masked_entropy_mse_region_2 = get_target_entropy_sme_masked(pwm_start=region_2_start, pwm_end=region_2_end, target_bits=region_2_target_bits)

        # pwm_sample_entropy_func_region_1 = get_pwm_margin_sample_entropy_masked(pwm_start=region_1_start, pwm_end=region_1_end, margin=similarity_margin, shift_1_nt=True)
        # pwm_sample_entropy_func_region_2 = get_pwm_margin_sample_entropy_masked(pwm_start=region_2_start, pwm_end=region_2_end, margin=similarity_margin, shift_1_nt=True)

        # def loss_func(loss_tensors) :
        #     _, _, _, sequence_class, pwm_logits_1, pwm_logits_2, pwm_1, pwm_2, sampled_pwm_1, sampled_pwm_2, mask, sampled_mask, hek_pred, hela_pred, mcf7_pred, cho_pred = loss_tensors

        #     #Create target isoform with sample axis
        #     iso_targets = K.constant(target_iso)
        #     iso_true = K.gather(iso_targets, sequence_class[:, 0])
        #     iso_true = K.tile(K.expand_dims(iso_true, axis=-1), (1, K.shape(sampled_pwm_1)[1], 1))

        #     #Specify costs
        #     iso_loss = 2.0 * K.mean(symmetric_sigmoid_kl_divergence(iso_true, hek_pred), axis=1)

        #     seq_loss = 0.0

        #     entropy_loss = entropy_weight * (masked_entropy_mse_region_1(pwm_1, mask) + masked_entropy_mse_region_2(pwm_1, mask)) / 2.
        #     entropy_loss += similarity_weight * (K.mean(pwm_sample_entropy_func_region_1(sampled_pwm_1, sampled_pwm_2, sampled_mask), axis=1) + K.mean(pwm_sample_entropy_func_region_2(sampled_pwm_1, sampled_pwm_2, sampled_mask), axis=1)) / 2.

        #     #Compute total loss
        #     total_loss = iso_loss + seq_loss + entropy_loss

        #     return total_loss

        # return loss_func
