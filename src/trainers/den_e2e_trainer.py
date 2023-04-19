import logging
from math import log
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Tuple, Union

import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import trange

from data.data_loader import create_dataloaders, get_oracle_splits
from data.old_dataset import create_sequence_templates
from models.base_cnn import OracleCNN
from models.den import DEN


class DenE2ETrainer:

    def __init__(self,
                 optimizer_type,
                 criterion,
                 oracle_save_path,
                 device: str,
                 save_dir: Union[str,
                                 Path],
                 save_plots: bool = True,
                 seed: int = 11202022,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every training run needs
        self.optimizer_type = optimizer_type
        self.criterion = criterion
        self.device = torch.device(device)
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

        embedding_template, embedding_mask = create_sequence_templates()
        self.den = DEN(embedding_template, embedding_mask, self.device, seq_length=101, n_classes=1).to(self.device)
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

        self.use_regularization = True

        self.name = 'den_e2e'

    def load_data(self) -> Tuple[DataLoader, DataLoader, any]:
        X_train, y_train, X_val, y_val = get_oracle_splits(seed=self.seed)
        self.train_loader, self.val_loader, self.data_dim = create_dataloaders(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, device=self.device)

    def save_model(self, name: str):
        torch.save(self.den.state_dict(), f'{self.save_dir}models/{name}.pt')

    def run_experiment(self):
        best_val_loss = 1e+5
        early_stopping_counter = 0

        for epoch in trange(1, self.epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()

            val_loss, spearman_res, (pearson_r, _) = self.eval(self.val_loader)

            log_string = (
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, spearman correlation: {spearman_res.correlation:.3f}, pearson correlation: {pearson_r:.3f}, patience:{early_stopping_counter},  "
                f"Epoch time = {(end_time - start_time):.3f}s")
            logging.info(log_string)

            if val_loss < best_val_loss:
                self.save_model(self.name)
                early_stopping_counter = 0
                best_val_loss = val_loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                break

    def eval(self, loader: DataLoader, save_plot=False) -> Tuple[torch.Tensor, object]:
        """evaluates model on given data loader. computes loss and spearman correlation between predictions and labels

        Args:
            loader (DataLoader): data loader containing validation/test data
            save_plot (bool, optional): whether or not to saved correlation plot. Defaults to False.

        Returns:
            torch.Tensor: validation loss
        """
        self.den.eval()

        predictions = torch.empty(loader.dataset.sequences.size(0)).to(self.device)
        full_labels: torch.Tensor = loader.dataset.proportions

        with torch.no_grad():
            for j, batch in enumerate(loader):
                examples, labels = batch

                predictions = self.den.trainable_predictor(examples).reshape(-1)

        if save_plot:
            # TODO: do plotting
            pass

        return self.criterion(predictions, full_labels.float()), spearmanr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy()), pearsonr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy())

    def train_epoch(self):
        self.den.train()
        running_loss = 0.0
        self.optimizer.zero_grad()

        sampled_pwm_1, predictions, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask = self.den()

        labels = self.oracle(sampled_pwm_1.reshape(-1, self.den.generator.seq_length, 4))

        # TODO: add similarity regularization
        loss = self.criterion(predictions, labels)

        if self.use_regularization:
            # diversity + entropy loss
            loss += self.get_reg_loss(sampled_pwm_1=sampled_pwm_1,
                                      sampled_pwm_2=sampled_pwm_2,
                                      pwm_1=pwm_1,
                                      onehot_mask=onehot_mask,
                                      sampled_onehot_mask=sampled_onehot_mask)

        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        return running_loss

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
