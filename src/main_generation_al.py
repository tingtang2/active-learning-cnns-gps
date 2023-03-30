import argparse
import logging
import sys
from datetime import date

import torch
from torch.nn import MSELoss
from torch.optim import AdamW, RMSprop

from models.base_cnn import BaseCNN
from models.dkl import GPRegressionModel
from trainers.exact_dkl_trainer import (ExactDKLDEIMOSTrainer,
                                        ExactDKLMaxVarTrainer,
                                        ExactDKLRandomTrainer)
from trainers.mc_dropout_trainer import (MCDropoutDEIMOSTrainer,
                                         MCDropoutMaxVarTrainer,
                                         MCDropoutRandomTrainer)

arg_model_trainer_map = {
    'random': (MCDropoutRandomTrainer,
               BaseCNN),
    'random_dkl': (ExactDKLRandomTrainer,
                   GPRegressionModel),
    'max_variance': (MCDropoutMaxVarTrainer,
                     BaseCNN),
    'max_variance_dkl': (ExactDKLMaxVarTrainer,
                         GPRegressionModel),
    'deimos_dkl': (ExactDKLDEIMOSTrainer,
                   GPRegressionModel),
    'deimos': (MCDropoutDEIMOSTrainer,
               BaseCNN)
}
arg_optimizer_map = {'rmsprop': RMSprop, 'adamw': AdamW}


def main() -> int:
    parser = argparse.ArgumentParser(description='Create and run active learning experiments on 5 prime splicing data')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs to train model')
    parser.add_argument('--device', '-d', default='cuda', type=str, help='cpu or gpu ID to use')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size used to train model')
    parser.add_argument('--num_acquisitions',
                        default=600,
                        type=int,
                        help='number of points to acquire per active learning experiment')
    parser.add_argument('--acquisition_batch_size',
                        default=1,
                        type=int,
                        help='number of points acquired at each active learning iteration')
    parser.add_argument('--pool_sample_size',
                        default=5000,
                        type=int,
                        help='number of points to sample to represent acquisition pool for max var strategy')
    parser.add_argument('--acquisition_dropout_iterations',
                        default=50,
                        type=int,
                        help='number of iterations to sample for mc dropout at inference during acquisition phase')
    parser.add_argument('--test_dropout_iterations',
                        default=200,
                        type=int,
                        help='number of iterations to sample for mc dropout at inference during test phase')
    parser.add_argument('--tau_inverse', default=0.15, type=float, help='const for DEIMOS strategy')
    parser.add_argument('--dropout_prob', default=0.15, type=float, help='probability for dropout before dense layers')
    parser.add_argument('--begin_train_set_size',
                        default=75,
                        type=int,
                        help='number of points to start active training experiment on')
    parser.add_argument('--l2_penalty', default=0.025, type=float, help='l2 penalty to start out with')
    parser.add_argument('--save_dir',
                        default='/gpfs/commons/home/tchen/al_project/active-learning-save/saved_metrics/',
                        help='path to saved metric files')
    parser.add_argument('--log_save_dir',
                        default='/gpfs/commons/home/tchen/al_project/active-learning-save/active-learning-logs/',
                        help='path to saved log files')
    parser.add_argument('--acquisition_fn_type', default='random', help='type of acquistion function to use')
    parser.add_argument('--optimizer', default='adamw', help='type of optimizer to use')
    parser.add_argument('--num_repeats', default=3, type=int, help='number of times to repeat experiment')
    parser.add_argument('--seed', default=11202022, type=int, help='random seed to be used in numpy and torch')
    parser.add_argument('--max_root_size',
                        default=20,
                        type=int,
                        help='max root decomposition size for predictive covariance in DEIMOS exact DKL method')

    args = parser.parse_args()
    configs = args.__dict__

    # for repeatability
    torch.manual_seed(configs['seed'])

    # set up logging
    filename = f'al-{configs["acquisition_fn_type"]}-{date.today()}-batch_size-{configs["acquisition_batch_size"]}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        filename=f'{configs["log_save_dir"]}{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    trainer_type, model_type = arg_model_trainer_map[configs['acquisition_fn_type']]
    trainer = trainer_type(model_type=model_type,
                           optimizer_type=arg_optimizer_map[configs['optimizer']],
                           criterion=MSELoss(),
                           **configs)

    # perform experiment n times
    for iter in range(configs['num_repeats']):
        trainer.load_data(iter)
        trainer.active_train_loop(iter)

    return 0


if __name__ == '__main__':
    sys.exit(main())
