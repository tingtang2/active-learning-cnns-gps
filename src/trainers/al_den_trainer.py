import logging
from math import ceil, log
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from data.data_loader import create_dataloaders, create_test_dataloader
from data.old_dataset import create_sequence_templates
from models.base_cnn import OracleCNN
from models.den import Generator
from trainers.base_trainer import BaseTrainer


class DenTrainer(BaseTrainer):

    def __init__(self, oracle_save_path, **kwargs) -> None:
        super().__init__(**kwargs)

        embedding_template, embedding_mask = create_sequence_templates()

        self.den = Generator(embedding_template,
                             embedding_mask,
                             self.device,
                             n_samples=10,
                             seq_length=101,
                             n_classes=1).to(self.device)
        self.optimizer = self.optimizer_type(self.den.parameters(), lr=self.learning_rate)

        self.oracle = OracleCNN().to(self.device)
        self.oracle.load_state_dict(torch.load(oracle_save_path))
        for param in self.oracle.parameters():
            param.requires_grad = False
        self.oracle.eval()

        # OLD CONFIGS TODO: FACTOR OUT
        self.region_1_start = 10
        self.region_1_end = 35
        self.region_1_target_bits = 2.0
        self.region_2_start = 53
        self.region_2_end = 78
        self.region_2_target_bits = 2.0
        self.entropy_weight = 3.5    #2.0,#1.0,
        self.similarity_weight = 7.5    #5.0,
        self.similarity_margin = 0.5

    def get_reg_loss(self, sampled_pwm_1, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask):
        entropy_loss = self.entropy_weight * (
            self.target_entropy_sme_masked(pwm=pwm_1,
                                           pwm_mask=onehot_mask,
                                           pwm_start=self.region_1_start,
                                           pwm_end=self.region_1_end,
                                           target_bits=self.region_1_target_bits) +
            self.target_entropy_sme_masked(pwm=pwm_1,
                                           pwm_mask=onehot_mask,
                                           pwm_start=self.region_2_start,
                                           pwm_end=self.region_2_end,
                                           target_bits=self.region_2_target_bits)) / 2.

        entropy_loss += self.similarity_weight * (
            torch.mean(self.pwm_margin_sample_entropy_masked(sampled_pwm_1=sampled_pwm_1,
                                                             sampled_pwm_2=sampled_pwm_2,
                                                             pwm_mask=sampled_onehot_mask,
                                                             pwm_start=self.region_1_start,
                                                             pwm_end=self.region_1_end,
                                                             margin=self.similarity_margin,
                                                             shift_1_nt=True),
                       dim=1) + torch.mean(self.pwm_margin_sample_entropy_masked(sampled_pwm_1=sampled_pwm_1,
                                                                                 sampled_pwm_2=sampled_pwm_2,
                                                                                 pwm_mask=sampled_onehot_mask,
                                                                                 pwm_start=self.region_2_start,
                                                                                 pwm_end=self.region_2_end,
                                                                                 margin=self.similarity_margin,
                                                                                 shift_1_nt=True),
                                           dim=1))
        return torch.sum(entropy_loss)

    def target_entropy_sme_masked(self, pwm, pwm_mask, pwm_start=0, pwm_end=100, target_bits=2.0, eps=1e-7):
        pwm_section = pwm[:, pwm_start:pwm_end, :, :]
        entropy = pwm_section * -torch.log(torch.clip(pwm_section, eps, 1. - eps)) / log(2.0)
        entropy = torch.sum(entropy, dim=(2, 3))
        conservation = 2.0 - entropy

        pwm_mask_section = pwm_mask[:, pwm_start:pwm_end, :, :]
        mask = torch.amax(pwm_mask_section, dim=(2, 3))
        n_unmasked = torch.sum(mask, dim=-1)

        return ((torch.sum(conservation * mask, dim=-1) / n_unmasked) - target_bits)**2

    def pwm_margin_sample_entropy_masked(self,
                                         sampled_pwm_1,
                                         sampled_pwm_2,
                                         pwm_mask,
                                         pwm_start=0,
                                         pwm_end=100,
                                         margin=0.5,
                                         shift_1_nt=False):
        sampled_pwm_1 = sampled_pwm_1[..., pwm_start:pwm_end, :, :]
        sampled_pwm_2 = sampled_pwm_2[..., pwm_start:pwm_end, :, :]

        sampled_pwm_mask = pwm_mask[..., pwm_start:pwm_end, :, :]
        mask = torch.amax(sampled_pwm_mask, dim=(-2, -1))
        n_unmasked = torch.sum(mask, dim=-1)

        mean_sample_ent = torch.sum(torch.sum(sampled_pwm_1 * sampled_pwm_2, dim=(-2, -1)) * mask, dim=-1) / n_unmasked
        mean_sample_ent_shift_l_1 = torch.sum(torch.sum(sampled_pwm_1[..., 1:, :, :] * sampled_pwm_2[..., :-1, :, :], dim=(-2, -1)) * mask[..., 1:], dim=-1) / n_unmasked
        mean_sample_ent_shift_r_1 = torch.sum(torch.sum(sampled_pwm_1[..., :-1, :, :] * sampled_pwm_2[..., 1:, :, :], dim=(-2, -1)) * mask[..., :-1], dim=-1) / n_unmasked

        margin_sample_ent = torch.where(mean_sample_ent > margin,
                                        mean_sample_ent - margin,
                                        torch.zeros_like(mean_sample_ent))
        margin_sample_ent_l_1 = torch.where(mean_sample_ent_shift_l_1 > margin,
                                            mean_sample_ent_shift_l_1 - margin,
                                            torch.zeros_like(mean_sample_ent))
        margin_sample_ent_r_1 = torch.where(mean_sample_ent_shift_r_1 > margin,
                                            mean_sample_ent_shift_r_1 - margin,
                                            torch.zeros_like(mean_sample_ent))

        if shift_1_nt:
            return margin_sample_ent + margin_sample_ent_l_1 + margin_sample_ent_r_1
        else:
            return margin_sample_ent

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')


class MCDropoutDenTrainer(DenTrainer):

    def active_train_loop(self, iter: int):
        # seed
        train_pool = [i for i in range(self.begin_train_set_size)]
        mse = []

        X_train_data = self.X_train[train_pool].astype(np.float32)
        y_train_data = self.y_train[train_pool].astype(np.float32)

        model = self.model_type(dropout_prob=self.dropout_prob).to(self.device)

        optimizer = self.optimizer_type(model.parameters(), lr=0.001, weight_decay=self.l2_penalty / len(train_pool))

        train_dataloader, test_dataloader, num_feats = create_dataloaders(X_train=X_train_data,
                                                                            y_train=y_train_data,
                                                                            X_test=self.X_test,
                                                                            y_test=self.y_test,
                                                                            device=self.device)
        new_mse, new_var = self.active_train_iteration(model=model,
                                                    train_loader=train_dataloader,
                                                    test_loader=test_dataloader,
                                                    optimizer=optimizer,
                                                    eval=True)

        mse.append(new_mse.item())
        logging.info(f'AL iteration: {0}, MSE: {mse[-1]}, len train set: {len(train_pool)}')

        sampled_pwm_1, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask = self.den()
        acquisition_pool_labels = self.oracle(sampled_pwm_1.reshape(-1, self.den.seq_length, 4))

        new_points, new_labels = self.acquisition_fn(pool_points=sampled_pwm_1, pool_labels=acquisition_pool_labels, model=model)

        synthetic_acquired_points = [(new_points, new_labels)]

        for round in trange(ceil(self.num_acquisitions / self.acquisition_batch_size)):
            model = self.model_type(dropout_prob=self.dropout_prob).to(self.device)

            optimizer = self.optimizer_type([{
                'params': model.parameters()
            },
                                             {
                                                 'params': self.den.parameters()
                                             }],
                                            lr=0.001,
                                            weight_decay=self.l2_penalty / len(train_pool))

            train_dataloader, test_dataloader, _ = create_dataloaders(X_train=X_train_data,
                                                                              y_train=y_train_data,
                                                                              X_test=self.X_test,
                                                                              y_test=self.y_test,
                                                                              device=self.device)

            # separate out training of normal data and generated data for convenient autograd purposes
            self.active_train_iteration(model=model,
                                        train_loader=train_dataloader,
                                        test_loader=test_dataloader,
                                        optimizer=optimizer,
                                        eval=False)

            self.train_synthetic_iteration(synthetic_pairs=synthetic_acquired_points, model=model, optimizer=optimizer)

            new_mse, new_var = self.eval(model=model, loader=create_test_dataloader(self.X_test, self.y_test, self.device), mc_dropout_iterations=self.test_dropout_iterations)

            mse.append(new_mse.item())
            logging.info(f'AL iteration: {round + 1}, MSE: {mse[-1]}, len train set: {len(train_pool)}')

            sampled_pwm_1, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask = self.den()

            acquisition_pool_labels = self.oracle(sampled_pwm_1.reshape(-1, self.den.seq_length, 4))
            new_points, new_labels = self.acquisition_fn(pool_points=sampled_pwm_1, pool_labels=acquisition_pool_labels, model=model)

            synthetic_acquired_points[-1][0].detach()
            synthetic_acquired_points[-1][1].detach()
            synthetic_acquired_points.append((new_points, new_labels))
            # free my homie
            model = None

        self.save_metrics(mse, iter)

    def train_synthetic_iteration(self, synthetic_pairs, model, optimizer):
        model.train()
        self.den.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in synthetic_pairs:
                optimizer.zero_grad()
                examples, labels = batch

                predictions = model(examples.reshape(-1, self.den.seq_length, 4)).reshape(-1)
                loss = self.criterion(predictions, labels.reshape(-1))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

    def active_train_iteration(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer,
        eval=True,
    ):
        model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                examples, labels = batch

                predictions = model(examples).reshape(-1)
                loss = self.criterion(predictions, labels)

                loss.backward(retain_graph=True)
                optimizer.step()

                running_loss += loss.item()

        if eval:
            return self.eval(model, test_loader, self.test_dropout_iterations)
        else:
            return

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


class MCDropoutMaxVarDenTrainer(MCDropoutDenTrainer):

    def __init__(self, **kwargs):
        super(MCDropoutMaxVarDenTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def acquisition_fn(self, pool_points, pool_labels, model) -> np.ndarray:
        pool_var = self.den_eval(model=model,
                                 pool_points=pool_points,
                                 pool_labels=pool_labels,
                                 mc_dropout_iterations=self.acquisition_dropout_iterations)

        return pool_points.reshape(-1, self.den.seq_length, 4)[torch.argsort(pool_var, descending=True)[:self.acquisition_batch_size]],  pool_labels[torch.argsort(pool_var, descending=True)[:self.acquisition_batch_size]]

    def den_eval(self,
                 model,
                 pool_points,
                 pool_labels,
                 mc_dropout_iterations: int) -> Tuple[torch.Tensor,
                                                      torch.Tensor]:
        model.eval()
        self._enable_dropout(model)

        predictions = torch.empty((mc_dropout_iterations, pool_labels.size(0))).to(self.device)

        with torch.no_grad():
            for i in range(mc_dropout_iterations):
                predictions[i] = model(pool_points.reshape(-1, self.den.seq_length, 4)).reshape(-1)
        pred_var, preds = torch.var_mean(predictions, dim=0)

        return pred_var