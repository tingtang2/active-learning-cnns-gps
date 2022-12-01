import logging
import math

import gpytorch
import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm, trange

from src.models.dkl import GPRegressionModel
from src.trainers.base_trainer import BaseTrainer


class ExactDKLTrainer(BaseTrainer):

    def active_train_loop(self, iter):
        train_pool = [i for i in range(self.begin_train_set_size)]
        acquisition_pool = [i for i in range(self.begin_train_set_size, self.X_train.shape[0])]
        mse = []

        for round in trange(math.ceil(self.num_acquisitions / self.acquisition_batch_size)):
            # hacky way to send the correctly batched data w/out gpytorch making a fuss
            X_train_data = torch.from_numpy(self.X_train[train_pool].astype(np.float32)).reshape(-1,
                                                                                                 404).float().to(
                                                                                                     self.device)
            y_train_data = torch.from_numpy(self.y_train[train_pool].astype(np.float32)).float().to(self.device)
            likelihood = GaussianLikelihood().to(self.device)
            model = GPRegressionModel(X_train_data,
                                      y_train_data,
                                      likelihood,
                                      dropout_prob=self.dropout_prob).to(self.device)

            # "Loss" for GPs - the marginal log likelihood
            mll = ExactMarginalLogLikelihood(likelihood, model)

            optimizer = self.optimizer_type([
                {
                    'params': model.feature_extractor.parameters()
                },
                {
                    'params': model.covar_module.parameters()
                },
                {
                    'params': model.mean_module.parameters()
                },
                {
                    'params': model.likelihood.parameters()
                },
            ],
                                            lr=0.001)

            new_mse, new_var = self.active_train_iteration(model, optimizer, mll, likelihood, X_train_data, y_train_data)

            mse.append(new_mse.item())

            logging.info(
                f'AL iteration: {round + 1}, MSE: {mse[-1]}, len train set: {len(train_pool)} len acquisition pool {len(acquisition_pool)}'
            )

            new_points = self.acquisition_fn(pool_points=acquisition_pool, model=model, likelihood=likelihood)

            for point in new_points:
                train_pool.append(point)
                acquisition_pool.remove(point)

        self.save_metrics(mse, iter)

    def active_train_iteration(self, model, optimizer, mll, likelihood, X_train_data, y_train_data):
        iterator = range(self.epochs)
        for i in iterator:
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(X_train_data)
            # Calc loss and backprop derivatives
            loss = -mll(output, y_train_data)
            loss.backward()

            # iterator.set_postfix(loss=loss.item())
            optimizer.step()

        return self.eval(model, likelihood, self.X_test, self.y_test)

    def eval(self, model, likelihood, test_X, test_y):
        model.eval()
        likelihood.eval()

        # hacky way to send the correctly batched data w/out gpytorch making a fuss
        X_test = torch.from_numpy(test_X).reshape(-1, 404).float().to(self.device)
        y_test = torch.from_numpy(test_y).float().to(self.device)

        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            preds = model(X_test)

        return torch.mean((preds.mean - y_test)**2), preds.variance


class ExactDKLMaxVarTrainer(ExactDKLTrainer):

    def __init__(self, **kwargs):
        super(ExactDKLMaxVarTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def acquisition_fn(self, pool_points, model, likelihood) -> np.ndarray:
        pool_sample = self.rng.choice(pool_points, self.pool_sample_size, replace=False)

        X_pool_data = self.X_train[pool_sample]
        y_pool_data = self.y_train[pool_sample]

        pool_mse, pool_var = self.eval(model=model, likelihood=likelihood, test_X=X_pool_data, test_y=y_pool_data)

        return pool_sample[torch.argsort(pool_var, descending=True)[:self.acquisition_batch_size].cpu().numpy()]
