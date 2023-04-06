import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Tuple, Union

import torch
from data.data_loader import create_dataloaders, get_oracle_splits
from data.old_dataset import create_sequence_templates
from models.base_cnn import OracleCNN
from models.den import DEN
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import trange


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

    def load_data(self) -> Tuple[DataLoader, DataLoader, any]:
        X_train, y_train, X_val, y_val = get_oracle_splits(seed=self.seed)
        self.train_loader, self.val_loader, self.data_dim = create_dataloaders(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, device=self.device)

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

        sampled_pwm, predictions = self.den()

        labels = self.oracle(sampled_pwm.reshape(-1, self.den.generator.seq_length, 4))
        loss = self.criterion(predictions, labels)

        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        return running_loss
