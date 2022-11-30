import logging
import math

import numpy as np
from tqdm import trange

from src.data.data_loader import create_dataloaders
from src.trainers.base_trainer import BaseTrainer
import torch
from torch.utils.data import DataLoader


class MCDropoutTrainer(BaseTrainer):

    def active_train_loop(self):
        train_pool = [i for i in range(self.begin_train_set_size)]
        acquisition_pool = [i for i in range(self.begin_train_set_size, self.X_train.shape[0])]
        mse = []

        for round in trange(math.ceil(self.num_acquisitions / self.acquisition_batch_size)):
            X_train_data = self.X_train[train_pool].astype(np.float32)
            y_train_data = self.y_train[train_pool].astype(np.float32)

            model = self.model_type().to(self.device)
            # extra insurance
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

            optimizer = self.optimizer_type(
                model.parameters(), lr=0.001, weight_decay=self.l2_penalty / len(train_pool)
            )

            train_dataloader, test_dataloader, num_feats = create_dataloaders(
                X_train=X_train_data, y_train=y_train_data, X_test=self.X_test, y_test=self.y_test, device=self.device
            )
            new_mse, new_var = self.active_iteration(
                model=model, train_loader=train_dataloader, test_loader=test_dataloader, optimizer=optimizer,
                criterion=self.criterion
            )

            mse.append(new_mse.item())
            logging.info(
                f'AL iteration: {round + 1}, MSE: {mse[-1]}, len train set: {len(train_pool)} len acquisition pool {len(acquisition_pool)}'
            )

            new_points = acquisition_fn(
                pool_points=acquisition_pool, X_train=self.X_train, y_train=self.y_train, model=model,
                criterion=self.criterion, device=self.device
            )

            for point in new_points:
                train_pool.append(point)
                acquisition_pool.remove(point)

    def active_train_iteration(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer,
        criterion,
    ):
        model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                examples, labels = batch

                predictions = model(examples).reshape(-1)

                loss = criterion(predictions, labels)

                loss.double().backward()
                optimizer.step()

                running_loss += loss.item()

        return self.eval(model, test_loader, criterion, self.mc_dropout_iterations, device=self.device)