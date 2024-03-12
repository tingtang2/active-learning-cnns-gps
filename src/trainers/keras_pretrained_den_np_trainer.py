import logging
from timeit import default_timer as timer
from typing import Tuple

import torch
from data.data_loader import create_dataloaders, get_oracle_splits
from models.np import SplicingConvCNP1d
from models.resnets import UNet
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import trange
from trainers.base_trainer import BaseTrainer
from models.keras_pretrained_den import KerasPretrainedDENPredictor, target_isos

from torch.distributions.normal import Normal


def gaussian_logpdf(inputs, mean, sigma, reduction=None):
    """Gaussian log-density.

    Args:
        inputs (tensor): Inputs.
        mean (tensor): Mean.
        sigma (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    dist = Normal(loc=mean, scale=sigma)
    logp = dist.log_prob(inputs)

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')


class PretrainedDenNpTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = SplicingConvCNP1d(inducer_net=UNet(in_channels=128), device=self.device).to(self.device)
        self.pretrained_den = KerasPretrainedDENPredictor(seed=self.seed, batch_size=32, path='')

        self.name = 'cnp_x_den'
        self.use_regularization = True

    def train_epoch(self, loader: DataLoader):
        self.den.train()
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            self.optimizer.zero_grad()
            true_examples, true_labels = batch

            # skip final leftover batch because problems will occur lol
            if true_examples.size(0) != self.batch_size:
                return running_loss

            sampled_pwm_1, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask = self.den()
            labels = self.oracle(sampled_pwm_1.reshape(-1, self.den.seq_length, 4))

            # switch context and targets?
            pred_dist = self.model(x_c=sampled_pwm_1.reshape(-1,
                                                             self.den.seq_length,
                                                             4),
                                   y_c=labels.to(self.device),
                                   x_t=true_examples.to(self.device))

            loss = -pred_dist.log_prob(true_labels).sum(-1).mean()

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
        self.X_train, self.y_train, self.X_val, self.y_val = get_oracle_splits(42, num=2)
        self.train_loader, self.val_loader, self.data_dim = create_dataloaders(X_train=self.X_train, y_train=self.y_train, X_test=self.X_val, y_test=self.y_val, device=self.device, batch_size=self.batch_size, test_batch_size=self.batch_size)

        best_val_loss = 1e+5
        early_stopping_counter = 0

        for epoch in trange(1, self.epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch(loader=self.train_loader)
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
        self.model.eval()

        predictions = []
        running_loss = 0

        with torch.no_grad():
            for j, batch in enumerate(loader):

                true_examples, true_labels = batch
                if true_examples.size(0) != self.batch_size:
                    continue

                sampled_pwm_1, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask = self.den()
                labels = self.oracle(sampled_pwm_1.reshape(-1, self.den.seq_length, 4))

                # switch context and targets?
                pred_dist = self.model(x_c=sampled_pwm_1.reshape(-1,
                                                                 self.den.seq_length,
                                                                 4),
                                       y_c=labels.to(self.device),
                                       x_t=true_examples.to(self.device))
                running_loss += -pred_dist.log_prob(true_labels).sum(-1).item()
                predictions.append(pred_dist.base_dist.loc)

        if save_plot:
            # TODO: do plotting
            pass

        predictions = torch.cat(predictions, dim=0).squeeze()
        full_labels: torch.Tensor = loader.dataset.proportions[:predictions.size(0)]

        # print('val loss:', running_loss / predictions.size(0))
        # print('full_labels', full_labels[:5], full_labels.dtype, full_labels.size())
        # print('predictions', predictions[:5], predictions.dtype, predictions.size())

        return running_loss/predictions.size(0), spearmanr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy()), pearsonr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy())