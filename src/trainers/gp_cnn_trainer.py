import math

import h5py
import scipy
import torch
from cnn_gp import save_K
from tqdm import trange

from src.models.infinite_filter import initialize_base_cnn
from src.trainers.base_trainer import BaseTrainer


class ConvNetGPTrainer(BaseTrainer):

    def solve_system(self, Kxx, Y):
        print("Running scipy solve Kxx^-1 Y routine")
        assert Kxx.dtype == torch.float64 and Y.dtype == torch.float64, """
        It is important that `Kxx` and `Y` are `float64`s for the inversion,
        even if they were `float32` when being calculated. This makes the
        inversion much less likely to complain about the matrix being singular.
        """
        A = scipy.linalg.solve(Kxx.numpy(),
                               Y.numpy(),
                               overwrite_a=True,
                               overwrite_b=False,
                               check_finite=False,
                               assume_a='pos',
                               lower=False)
        return torch.from_numpy(A)

    def active_train_loop(self, iter):
        train_pool = [i for i in range(self.begin_train_set_size)]
        acquisition_pool = [i for i in range(self.begin_train_set_size, self.X_train.shape[0])]
        mse = []

        for round in trange(math.ceil(self.num_acquisitions / self.acquisition_batch_size)):
            model = initialize_base_cnn()
            # hacky way to send the correctly batched data w/out gpytorch making a fuss
            X_train_data = torch.from_numpy(self.X_train[train_pool].astype(np.float32)).reshape(-1,
                                                                                                 404).float().to(
                                                                                                     self.device)
            y_train_data = torch.from_numpy(self.y_train[train_pool].astype(np.float32)).float().to(self.device)

            self.active_train_iteration(model,)

    def active_train_iteration(self, model, X_train_data, y_train_data):

        def kern(x, x2, same, diag):
            with torch.no_grad():
                return model(x.cuda(), x2.cuda(), same, diag).detach().cpu().numpy()

        with h5py.File(self.save_dir, "w") as f:
            kwargs = dict(worker_rank=0, n_workers=1, batch_size=self.batch_size, print_interval=2.)
            save_K(f, kern, name="Kxx", X=X_train_data, X2=None, diag=False, **kwargs)
            save_K(f, kern, name="Kxvx", X=dataset.validation, X2=dataset.train, diag=False, **kwargs)
            save_K(f, kern, name="Kxtx", X=dataset.test, X2=dataset.train, diag=False, **kwargs)
            save_K(f, kern, name="Kv_diag", X=dataset.validation, X2=None, diag=True, **kwargs)
            save_K(f, kern, name="Kt_diag", X=dataset.test, X2=None, diag=True, **kwargs)

    def eval(self):
        return super().eval()