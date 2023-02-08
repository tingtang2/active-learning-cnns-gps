import json
import logging
import math
from pathlib import Path
from typing import List

import gpytorch
import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm, trange

from models.dkl import GPRegressionModel
from trainers.base_trainer import BaseTrainer


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

            new_points = self.acquisition_fn(pool_points=acquisition_pool,
                                             train_points=train_pool,
                                             model=model,
                                             likelihood=likelihood)

            for point in new_points:
                train_pool.append(point)
                acquisition_pool.remove(point)

        self.save_metrics(mse, iter)

    def active_train_iteration(self, model, optimizer, mll, likelihood, X_train_data, y_train_data):
        model.train()
        likelihood.train()
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

        with torch.no_grad():
            preds = model(X_test)

        return torch.mean((preds.mean - y_test)**2), preds.variance


class ExactDKLRandomTrainer(ExactDKLTrainer):

    def __init__(self, **kwargs):
        super(ExactDKLRandomTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def save_metrics(self, metrics: List[float], iter: int):
        save_name = f'{self.acquisition_fn_type}_iteration_{iter}-batch_size-{self.acquisition_batch_size}-refactor-num-acquisitions-{self.num_acquisitions}-direct-posterior-exact.json'
        with open(Path(Path.home(), self.save_dir, save_name), 'w') as f:
            json.dump(metrics, f)

    def acquisition_fn(self, pool_points, train_points, model, likelihood) -> np.ndarray:
        return self.rng.choice(np.array(pool_points), self.acquisition_batch_size, replace=False)


class ExactDKLMaxVarTrainer(ExactDKLTrainer):

    def __init__(self, **kwargs):
        super(ExactDKLMaxVarTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def save_metrics(self, metrics: List[float], iter: int):
        save_name = f'{self.acquisition_fn_type}_iteration_{iter}-batch_size-{self.acquisition_batch_size}-refactor-num-acquisitions-{self.num_acquisitions}-exact.json'
        with open(Path(Path.home(), self.save_dir, save_name), 'w') as f:
            json.dump(metrics, f)

    def acquisition_fn(self, pool_points, train_points, model, likelihood) -> np.ndarray:
        pool_sample = self.rng.choice(pool_points, self.pool_sample_size, replace=False)

        X_pool_data = self.X_train[pool_sample]
        y_pool_data = self.y_train[pool_sample]

        pool_mse, pool_var = self.eval(model=model, likelihood=likelihood, test_X=X_pool_data, test_y=y_pool_data)

        return pool_sample[torch.argsort(pool_var, descending=True)[:self.acquisition_batch_size].cpu().numpy()]


class ExactDKLDEIMOSTrainer(ExactDKLTrainer):

    def __init__(self, **kwargs):
        super(ExactDKLDEIMOSTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def save_metrics(self, metrics: List[float], iter: int):
        #save_name = f'{self.acquisition_fn_type}_iteration_{iter}-batch_size-{self.acquisition_batch_size}-refactor-num-acquisitions-{self.num_acquisitions}-direct-posterior-max-root-size-{self.max_root_size}.json'
        save_name = f'{self.acquisition_fn_type}_iteration_{iter}-batch_size-{self.acquisition_batch_size}-refactor-num-acquisitions-{self.num_acquisitions}-exact.json'
        with open(Path(Path.home(), self.save_dir, save_name), 'w') as f:
            json.dump(metrics, f)

    def eval_for_covar(self, model, likelihood, test_X):
        model.eval()
        likelihood.eval()

        # hacky way to send the correctly batched data w/out gpytorch making a fuss
        X_test = torch.from_numpy(test_X).reshape(-1, 404).float().to(self.device)

        with torch.no_grad():
            preds = likelihood(model(X_test))
            fast_covar = preds.covariance_matrix

        return fast_covar

    def acquisition_fn(self, pool_points, train_points, model, likelihood) -> np.ndarray:
        # TODO: clean this logic up...

        # get sub indicies of concatted pool + train points
        sample_indicies = self.rng.choice(len(train_points) + len(pool_points),
                                          int(1.2 * self.pool_sample_size),
                                          replace=False)

        # ratio of sampled points that are within the pool
        pool_sample_ratio = len(sample_indicies[sample_indicies >= len(train_points)]) / float(self.pool_sample_size)
        train_sample_size = len(sample_indicies[sample_indicies < len(train_points)])

        # get relative number of training points selected with respect to pool sample size
        sample_indicies_train = (sample_indicies[sample_indicies < len(train_points)])[0:int(train_sample_size //
                                                                                             pool_sample_ratio)]
        sample_indicies_pool = (sample_indicies[sample_indicies >= len(train_points)])[0:self.pool_sample_size]

        # set up data for GP inference
        X_ei_train_data = self.X_train[train_points][sample_indicies_train]
        X_ei_pool_data = np.concatenate((self.X_train[train_points], self.X_train[pool_points]))[sample_indicies_pool]
        X_ei_concat = np.concatenate((X_ei_train_data, X_ei_pool_data), axis=0)

        # Running EI acquisition
        pred_covar = self.eval_for_covar(model, likelihood, X_ei_concat)

        acq_fn_results = self._run_ei_acquisition(pred_covar.detach().cpu().numpy(),
                                                  len(sample_indicies_pool),
                                                  len(sample_indicies_train))
        acq_ind_ind = sample_indicies_pool[acq_fn_results] - len(train_points)    # get correct index
        acq_ind = pool_points[acq_ind_ind]
        return [acq_ind]

    def _run_ei_acquisition(self, univ_covariance, num_pool_samples, num_training_samples):
        """Given Var(Y_{sample}), applies batch-mode EI active learning to query points

            CURRENTLY TURNING OFF BATCH MODE FOR COMPUTATIONAL REASONS

            # Arguments
                univ_covariance: Var(Y_{sample})
                num_pool_samples: number of pool points in D_{sample}
                num_training_samples: number of training points in D_{sample}
                batch_size: number of queried points per batch
            # Returns
                the indicies of queried pool points as they are arranged in univ_covariance
            """

        all_acq_values = np.zeros(num_pool_samples)

        for new_pt_ind in range(num_pool_samples):
            # Cov_q(Y^hat_samp (theta), y^hat_new (theta) )
            covariance_vector = univ_covariance[num_training_samples + new_pt_ind, :]

            # sum of the square values equals the trace of the covariance
            all_acq_values[new_pt_ind] = np.sum(
                np.square(covariance_vector)) / (univ_covariance[num_training_samples + new_pt_ind,
                                                                 num_training_samples + new_pt_ind])

        return np.argsort(all_acq_values)[-1]
