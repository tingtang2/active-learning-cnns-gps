import logging
import math
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from src.data.data_loader import create_dataloaders, create_test_dataloader
from src.trainers.base_trainer import BaseTrainer


class MCDropoutTrainer(BaseTrainer):

    def active_train_loop(self, iter: int):
        train_pool = [i for i in range(self.begin_train_set_size)]
        acquisition_pool = [i for i in range(self.begin_train_set_size, self.X_train.shape[0])]
        mse = []

        for round in trange(math.ceil(self.num_acquisitions / self.acquisition_batch_size)):
            X_train_data = self.X_train[train_pool].astype(np.float32)
            y_train_data = self.y_train[train_pool].astype(np.float32)

            model = self.model_type().to(self.device)

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

            new_points = self.acquisition_fn(pool_points=acquisition_pool, model=model)

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

        return self.eval(model, test_loader)

    def _enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def eval(self, model, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        self._enable_dropout(model)

        predictions = torch.empty((self.mc_dropout_iterations, loader.dataset.sequences.size(0))).to(self.device)
        full_labels: torch.Tensor = loader.dataset.proportions

        with torch.no_grad():
            for i in range(self.mc_dropout_iterations):
                for j, batch in enumerate(loader):
                    examples, labels = batch

                    predictions[i] = model(examples).reshape(-1)
        pred_var, preds = torch.var_mean(predictions, dim=0)

        return self.criterion(preds, full_labels.float()), pred_var


class MCDropoutRandomTrainer(MCDropoutTrainer):

    def __init__(self, **kwargs):
        super(MCDropoutRandomTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def acquisition_fn(self, pool_points, model) -> np.ndarray:
        return self.rng.choice(np.array(pool_points), self.acquisition_batch_size, replace=False)


class MCDropoutMaxVarTrainer(MCDropoutTrainer):

    def __init__(self, **kwargs):
        super(MCDropoutMaxVarTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def acquisition_fn(self, pool_points, model) -> np.ndarray:
        pool_sample = self.rng.choice(pool_points, self.pool_sample_size, replace=False)

        X_pool_data = self.X_train[pool_sample]
        y_pool_data = self.y_train[pool_sample]

        pool_dataloader = create_test_dataloader(X_pool_data, y_pool_data, self.device)
        pool_mse, pool_var = self.eval(model=model, loader=pool_dataloader)

        return pool_sample[torch.argsort(pool_var, descending=True)[:self.acquisition_batch_size].cpu().numpy()]