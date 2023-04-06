import sys
import argparse

import torch
import logging

from torch.optim import AdamW, RMSprop
from datetime import date
from torch.nn import MSELoss
from trainers.den_e2e_trainer import DenE2ETrainer

arg_optimizer_map = {'rmsprop': RMSprop, 'adamw': AdamW}


def main() -> int:
    parser = argparse.ArgumentParser(description='Create and run oracle learning experiments on 5 prime splicing data')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train model')
    parser.add_argument('--device', '-d', default='cuda', type=str, help='cpu or gpu ID to use')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size used to train model')
    parser.add_argument('--dropout_prob', default=0.15, type=float, help='probability for dropout before dense layers')
    parser.add_argument('--l2_penalty', default=0.025, type=float, help='l2 penalty to start out with')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate for optimizer')
    parser.add_argument('--early_stopping_threshold',
                        default=10,
                        type=int,
                        help='threshold for when val loss stops decreasing')
    parser.add_argument('--save_dir',
                        default='/gpfs/commons/home/tchen/al_project/active-learning-save/saved_metrics/',
                        help='path to saved metric files')
    parser.add_argument('--log_save_dir',
                        default='/gpfs/commons/home/tchen/al_project/active-learning-save/active-learning-logs/',
                        help='path to saved log files')
    parser.add_argument(
        '--oracle_save_path',
        default='/gpfs/commons/home/tchen/al_project/active-learning-save/saved_metrics/models/base_cnn_oracle.pt',
        help='path to saved oracle pt file')
    parser.add_argument('--model_type', default='cnn', help='type of model to use')
    parser.add_argument('--optimizer', default='adamw', help='type of optimizer to use')
    parser.add_argument('--num_repeats', default=1, type=int, help='number of times to repeat experiment')
    parser.add_argument('--seed', default=11202022, type=int, help='random seed to be used in numpy and torch')
    parser.add_argument('--turn_off_wandb', action='store_true', help='skip wandb logging')

    args = parser.parse_args()
    configs = args.__dict__

    # for repeatability
    torch.manual_seed(configs['seed'])

    # set up logging
    filename = f'den-e2e-{date.today()}'
    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        filename=f'{configs["log_save_dir"]}{filename}.log',
                        filemode='a',
                        format=FORMAT)
    logging.info(configs)

    # get trainer
    trainer_type = DenE2ETrainer
    trainer = trainer_type(optimizer_type=arg_optimizer_map[configs['optimizer']], criterion=MSELoss(), **configs)

    # perform experiment n times
    for iter in range(configs['num_repeats']):
        trainer.load_data()
        trainer.run_experiment()
    return 0


if __name__ == '__main__':
    sys.exit(main())