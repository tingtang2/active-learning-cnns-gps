from trainers.al_den_trainer import DenTrainer
from models.np import ConvCNP1d
import logging

from tqdm import trange
from typing import Tuple

from timeit import default_timer as timer
import torch

from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader


class NpDenTrainer(DenTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = ConvCNP1d()

        self.name = 'cnp_x_den'

    def train_epoch(self):
        self.den.train()
        running_loss = 0.0
        self.optimizer.zero_grad()

        sampled_pwm_1, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask = self.den()
        labels = self.oracle(sampled_pwm_1.reshape(-1, self.den.generator.seq_length, 4))

        # switch context and targets?
        pred_dist = self.model(xc=self.X_train, yc=self.y_train, xt=sampled_pwm_1)

        loss = -pred_dist.log_prob(labels).sum(-1).mean()

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

    def run_experiment(self):
        self.load_data(2)

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

                predictions = self.model(examples).reshape(-1)

        if save_plot:
            # TODO: do plotting
            pass

        return self.criterion(predictions, full_labels.float()), spearmanr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy()), pearsonr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy())