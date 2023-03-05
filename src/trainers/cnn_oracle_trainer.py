import logging
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import trange

from models.base_cnn import BaseCNN
from trainers.base_trainer import BaseOracleTrainer


class CNNOracleTrainer(BaseOracleTrainer):

    def __init__(self, **kwargs) -> None:
        super(CNNOracleTrainer, self).__init__(**kwargs)

        self.model = BaseCNN(dropout_prob=self.dropout_prob).to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.learning_rate)

        self.name = 'base_cnn_oracle'

    def run_experiment(self):
        best_val_loss = 1e+5
        early_stopping_counter = 0

        for epoch in trange(1, self.epochs + 1):
            start_time = timer()
            train_loss = self.train_epoch(self.train_loader)
            end_time = timer()

            val_loss, sig_result = self.eval(self.val_loader)

            logging.info((
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Spearman correlation: {sig_result.statistic:.3f}, pvalue: {sig_result.pvalue:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))

            if val_loss < best_val_loss:
                self.save_model(self.name)
                logging.info('saving model')
                early_stopping_counter = 0
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
        self.model.eval()

        predictions = torch.empty(loader.dataset.sequences.size(0)).to(self.device)
        full_labels: torch.Tensor = loader.dataset.proportions

        with torch.no_grad():
            for j, batch in enumerate(loader):
                examples, labels = batch

                predictions = self.model(examples).reshape(-1)

        if save_plot:
            # TODO: do plotting
            pass

        return self.criterion(predictions, full_labels.float()), spearmanr(predictions.detach().cpu().numpy(), full_labels.detach().cpu().numpy())

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            examples, labels = batch

            predictions = self.model(examples).reshape(-1)
            loss = self.criterion(predictions, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        return running_loss