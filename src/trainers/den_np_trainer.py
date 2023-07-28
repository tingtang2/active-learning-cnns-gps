from trainers.al_den_trainer import DenTrainer
from models.np import ConvCNP1d


class NpDenTrainer(DenTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = ConvCNP1d()

    def train_epoch(self):
        self.den.train()
        running_loss = 0.0
        self.optimizer.zero_grad()

        sampled_pwm_1, sampled_pwm_2, pwm_1, onehot_mask, sampled_onehot_mask = self.den()

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

    def train_synthetic_iteration(self, synthetic_pairs, optimizer):
        self.model.train()
        self.den.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in synthetic_pairs:
                optimizer.zero_grad()
                examples, labels = batch

                predictions = model(examples.reshape(-1, self.den.seq_length, 4)).reshape(-1)
                loss = self.criterion(predictions, labels.reshape(-1))

                loss.backward(retain_graph=True)
                optimizer.step()

                running_loss += loss.item()

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

        for round in trange(ceil(self.num_acquisitions / self.acquisition_batch_size)):
            model = self.model_type(dropout_prob=self.dropout_prob).to(self.device)

            optimizer = self.optimizer_type([{
                'params': model.parameters()
            },
                                             {
                                                 'params': self.den.parameters()
                                             }],
                                            lr=0.001,
                                            weight_decay=self.l2_penalty / len(train_pool))

            train_dataloader, test_dataloader, _ = create_dataloaders(X_train=X_train_data,
                                                                              y_train=y_train_data,
                                                                              X_test=self.X_test,
                                                                              y_test=self.y_test,
                                                                              device=self.device)

            # separate out training of normal data and generated data for convenient autograd purposes
            self.active_train_iteration(model=model,
                                        train_loader=train_dataloader,
                                        test_loader=test_dataloader,
                                        optimizer=optimizer,
                                        eval=False)

            self.train_synthetic_iteration(synthetic_pairs=synthetic_acquired_points, model=model, optimizer=optimizer)

            new_mse, new_var = self.eval(model=model, loader=create_test_dataloader(self.X_test, self.y_test, self.device), mc_dropout_iterations=self.test_dropout_iterations)