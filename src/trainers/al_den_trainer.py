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
    def active_train_loop(self, iter: int):
        train_pool = [i for i in range(self.begin_train_set_size)]
        acquisition_pool = [i for i in range(self.begin_train_set_size, self.X_train.shape[0])]
        mse = []

        for round in trange(math.ceil(self.num_acquisitions / self.acquisition_batch_size)):
            X_train_data = self.X_train[train_pool].astype(np.float32)
            y_train_data = self.y_train[train_pool].astype(np.float32)

            model = self.model_type(dropout_prob=self.dropout_prob).to(self.device)

            optimizer = self.optimizer_type(model.parameters(),
                                            lr=0.001,
                                            weight_decay=self.l2_penalty / len(train_pool))

            train_dataloader, test_dataloader, num_feats = create_dataloaders(X_train=X_train_data,
                                                                              y_train=y_train_data,
                                                                              X_test=self.X_test,
                                                                              y_test=self.y_test,
                                                                              device=self.device)
            new_mse, new_var = self.active_train_iteration(model=model,
                                                     train_loader=train_dataloader,
                                                     test_loader=test_dataloader,
                                                     optimizer=optimizer)

            mse.append(new_mse.item())
            logging.info(
                f'AL iteration: {round + 1}, MSE: {mse[-1]}, len train set: {len(train_pool)} len acquisition pool {len(acquisition_pool)}'
            )

            new_points = self.acquisition_fn(pool_points=acquisition_pool, train_points=train_pool, model=model)

            for point in new_points:
                train_pool.append(point)
                acquisition_pool.remove(point)

        self.save_metrics(mse, iter)

    def active_train_iteration(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer,
    ):
        model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                examples, labels = batch

                predictions = model(examples).reshape(-1)
                loss = self.criterion(predictions, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        return self.eval(model, test_loader, self.test_dropout_iterations)

    def _enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def eval(self, model, loader: DataLoader, mc_dropout_iterations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        self._enable_dropout(model)

        predictions = torch.empty((mc_dropout_iterations, loader.dataset.sequences.size(0))).to(self.device)
        full_labels: torch.Tensor = loader.dataset.proportions

        with torch.no_grad():
            for i in range(mc_dropout_iterations):
                for j, batch in enumerate(loader):
                    examples, labels = batch

                    predictions[i] = model(examples).reshape(-1)
        pred_var, preds = torch.var_mean(predictions, dim=0)

        return self.criterion(preds, full_labels.float()), pred_var

    def acquisition_fn(self, pool_points, train_points, model) -> np.ndarray:
        return self.rng.choice(np.array(pool_points), self.acquisition_batch_size, replace=False)
