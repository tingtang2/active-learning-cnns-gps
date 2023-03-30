# contains modules for deep kernel learning baselines
import gpytorch

from models.base_cnn import BaseCNN


class GPRegressionModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, dropout_prob):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                                                                     num_dims=2,
                                                                     grid_size=100)
        self.feature_extractor = BaseCNN(dropout_prob=dropout_prob, output_dim=2)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)    # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return mvn


# TODO: Figure this stuff out
# # approximate GP
# class GaussianProcessLayer(gpytorch.models.ApproximateGP):

#     def __init__(self, num_dim=101, grid_bounds=(-10., 10.), grid_size=64):
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=grid_size,
#                                                                                         batch_shape=torch.Size(
#                                                                                             [num_dim]))

#         # Our base variational strategy is a GridInterpolationVariationalStrategy,
#         # which places variational inducing points on a Grid
#         # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
#         variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
#             gpytorch.variational.GridInterpolationVariationalStrategy(
#                 self,
#                 grid_size=grid_size,
#                 grid_bounds=[grid_bounds],
#                 variational_distribution=variational_distribution,
#             ),
#             num_tasks=num_dim,
#         )
#         super().__init__(variational_strategy)

#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(math.exp(-1),
#                                                                                           math.exp(1),
#                                                                                           sigma=0.1,
#                                                                                           transform=torch.exp)))
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.grid_bounds = grid_bounds

#     def forward(self, x):
#         mean = self.mean_module(x)
#         covar = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)

# class ApproximateDKLRegression(gpytorch.Module):

#     def __init__(self, num_dim=101, grid_bounds=(-10., 10.)):
#         super(ApproximateDKLRegression, self).__init__()
#         self.feature_extractor = BaseCNN()
#         self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
#         self.grid_bounds = grid_bounds
#         self.num_dim = num_dim

#         # This module will scale the NN features so that they're nice values
#         self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         features = self.scale_to_bounds(features)
#         # This next line makes it so that we learn a GP for each feature
#         features = features.transpose(-1, -2).unsqueeze(-1)
#         res = self.gp_layer(features)
#         return res