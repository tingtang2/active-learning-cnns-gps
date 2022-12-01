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

        return self.eval(model, test_loader, self.test_dropout_iterations)

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


class MCDropoutRandomTrainer(MCDropoutTrainer):

    def __init__(self, **kwargs):
        super(MCDropoutRandomTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def acquisition_fn(self, pool_points, train_points, model) -> np.ndarray:
        return self.rng.choice(np.array(pool_points), self.acquisition_batch_size, replace=False)


class MCDropoutMaxVarTrainer(MCDropoutTrainer):

    def __init__(self, **kwargs):
        super(MCDropoutMaxVarTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def acquisition_fn(self, pool_points, train_points, model) -> np.ndarray:
        pool_sample = self.rng.choice(pool_points, self.pool_sample_size, replace=False)

        X_pool_data = self.X_train[pool_sample]
        y_pool_data = self.y_train[pool_sample]

        pool_dataloader = create_test_dataloader(X_pool_data, y_pool_data, self.device)
        pool_mse, pool_var = self.eval(model=model, loader=pool_dataloader, mc_dropout_iterations=self.acquisition_dropout_iterations)

        return pool_sample[torch.argsort(pool_var, descending=True)[:self.acquisition_batch_size].cpu().numpy()]


class MCDropoutDEIMOSTrainer(MCDropoutTrainer):

    def __init__(self, **kwargs):
        super(MCDropoutMaxVarTrainer, self).__init__(**kwargs)

        self.rng = np.random.default_rng(self.seed)

    def acquisition_fn(self, pool_points, train_points, model) -> np.ndarray:
        # TODO: clean this logic up...

        # get sub indicies of concatted pool + train points
        sample_indices = self.rng.choice(len(train_points) + len(pool_points),
                                         int(1.2 * self.pool_sample_size),
                                         replace=False)

        # ratio of sampled points that are within the pool
        pool_sample_ratio = len(sample_indices[sample_indices >= len(train_points)]) / self.pool_sample_size
        train_sample_size = len(sample_indices[sample_indices < len(train_points)])

        # get relative number of training points selected with respect to pool sample size
        sample_indices_train = (sample_indices[sample_indices < len(train_points)])[0:int(train_sample_size //
                                                                                          pool_sample_ratio)]
        sample_indices_pool = (sample_indices[sample_indices >= len(train_points)])[0:self.pool_sample_size]

        X_ei_train_data = self.X_train[train_points][sample_indices_train]
        X_ei_pool_data = np.concatenate((self.X_train[train_points], self.X_train[pool_points]))[sample_indices_pool]

        # Running EI acquisition
        acq_fn_results = self._get_acquisition_fn(model,
                                                  X_train_fn,
                                                  X_pool_fn,
                                                  acq_fn_dropout_iterations,
                                                  tau_inverse,
                                                  batch_size,
                                                  dropout_prob)
        acq_fn_ind = acq_fn_results
        acq_ind_ind = np.subtract(sample_indices_pool[acq_fn_ind], len(train_points))
        acq_ind = np.array(pool_points)[acq_ind_ind]
        return acq_ind

    def _get_acquisition_fn(self, model, X_train_sample, X_cand, num_masks, tau_inverse, batch_size, dropout_prob):
        """Given sample points, generates J fixed-mask predictions across all sample points, calculates Var(Y_{sample}), and runs batch-mode EI acquisition
            #Arguments
            model: keras model
            X_train_sample: numpy array of sample points that are in the training set
            X_cand: numpy array of sample points that are in the pool
            num_masks: J, the number of dropout masks being used in estimation of Var(Y_{sample}) and calculation of the EI acquisition function
            tau_inverse: inverse model precision hyperparameter
            batch_size: number of queried points per batch
            dropout_prob: dropout probability 
            #Returns
            The indices of pool points queried by batch-mode EI as they are arranged in X_{cand}
        """

        forward_pass_input = np.concatenate((X_train_sample, X_cand))
        forward_pass_output = np.zeros((len(X_train_sample) + len(X_cand), num_masks))
        #Generating fixed dropout masks
        conv_masks = 1 / (1 - dropout_prob) * np.random.choice(2,
                                                               size=((num_masks,
                                                                      1,
                                                                      9,
                                                                      50)),
                                                               p=[dropout_prob,
                                                                  1 - dropout_prob])
        dense_masks = 1 / (1 - dropout_prob) * np.random.choice(2,
                                                                size=((num_masks,
                                                                       1,
                                                                       1,
                                                                       50)),
                                                                p=[dropout_prob,
                                                                   1 - dropout_prob])
        #Generating fixed mask predictions in 2000 point chunks (to avoid exceeding memory limits)
        last_point_ind = 0
        while last_point_ind < len(X_train_sample) + len(X_cand):
            if last_point_ind + 2000 < len(X_train_sample) + len(X_cand):
                forward_pass_chunk = forward_pass_input[last_point_ind:last_point_ind + 2000]
                forward_pass_output[last_point_ind:last_point_ind + 2000] = self._fixed_mask_forward_pass(
                    model,
                    forward_pass_chunk,
                    num_masks,
                    dropout_prob,
                    conv_masks,
                    dense_masks).T
            else:
                forward_pass_chunk = forward_pass_input[last_point_ind:]
                forward_pass_output[last_point_ind:] = self._fixed_mask_forward_pass(
                    model,
                    forward_pass_chunk,
                    num_masks,
                    dropout_prob,
                    conv_masks,
                    dense_masks).T
            last_point_ind += 2000
        output_covariance = np.cov(forward_pass_output)
        print('num training: ' + str(len(X_train_sample)) + ', num cand: ' + str(len(X_cand)))
        print('Tau Inverse Value: ' + str(tau_inverse))
        #Var(Y_{sample}) = Var(\hat{Y}_{sample}) + tau^{-1} I
        final_output_covariance = output_covariance + (tau_inverse * np.identity(output_covariance.shape[0]))
        return self.ei_acquisition_fn_model_var(final_output_covariance, len(X_cand), len(X_train_sample), batch_size)

    def _ei_acquisition_fn_model_var(self, univ_covariance, num_pool_samples, num_training_samples, batch_size):
        """Given Var(Y_{sample}), applies batch-mode EI active learning to query points
        #Arguments
            univ_covariance: Var(Y_{sample})
            num_pool_samples: number of pool points in D_{sample}
            num_training_samples: number of training points in D_{sample}
            batch_size: number of queried points per batch
        #Returns
            the indices of queried pool points as they are arranged in univ_covariance
        """
        acq_ind = []
        for acq_num in range(batch_size):
            all_acq_values = np.zeros(num_pool_samples)
            for new_pt_ind in range(num_pool_samples):
                covariance_vector = univ_covariance[num_training_samples + new_pt_ind, :]
                all_acq_values[new_pt_ind] = np.sum(
                    np.square(covariance_vector)) / (univ_covariance[num_training_samples + new_pt_ind,
                                                                     num_training_samples + new_pt_ind])
            sorted_top_ind = np.flip(np.argsort(all_acq_values))
            found_new_ind = False
            top_ind_ctr = -1
            while (found_new_ind == False):
                top_ind_ctr += 1
                new_top_ind = sorted_top_ind[top_ind_ctr]
                if new_top_ind not in acq_ind:
                    acq_ind.append(new_top_ind)
                    found_new_ind = True
            top_cov_vector = np.expand_dims(univ_covariance[num_training_samples + acq_ind[-1], :], axis=1)
            univ_covariance = univ_covariance - np.matmul(
                top_cov_vector,
                top_cov_vector.T) / univ_covariance[num_training_samples + acq_ind[-1],
                                                    num_training_samples + acq_ind[-1]]
        return acq_ind

    def _fixed_mask_forward_pass(self, model, forward_pass_input, num_masks, dropout_prob, conv_masks, dense_masks):
        """Makes model predictions with J dropout masks that are fixed across points to enable estimation of Var(Y_{sample})
        Function is specific to the given Keras model.
        #Arguments
            model: keras model
            forward_pass_input: X_{sample}
            num_masks: J, the number of dropout masks being used in estimation of Var(Y_{sample}) and calculation of the EI acquisition function
            dropout_prob: dropout probability 
            conv_masks: first set of dropout masks, applied after second MaxPooling1D layer
            dense_masks: second set of dropout masks, applied after the first Dense layer
        #Returns
            MC dropout predictions across sample points enabling estimation of Var(Y_{sample}), i.e. \hat{Y}_{sample}
        """
        # Functions to retrieve output of intermediate layers
        # Needed for manual implementation of fixed dropout masks
        # across all data points
        conv = K.function([model.layers[0].input, K.learning_phase()], [model.layers[4].output])

        dense_1 = K.function([model.layers[6].input, K.learning_phase()], [model.layers[7].output])

        dense_2 = K.function([model.layers[9].input, K.learning_phase()], [model.layers[9].output])

        conv_output = np.array(conv((forward_pass_input, 1)))
        dense_1_input = apply_dropout_masks(conv_output, conv_masks)
        dense_1_output = multi_mask_predict(dense_1, dense_1_input)
        dense_2_input = apply_dropout_masks(dense_1_output, dense_masks)
        dense_2_output = np.squeeze(multi_mask_predict(dense_2, dense_2_input))
        return dense_2_output
