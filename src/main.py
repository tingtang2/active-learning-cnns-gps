import sys
import logging
from datetime import date

def main() -> int:
    configs = {
        'epochs': 300,
        'batch_size': 128,
        'num_acquisitions': 600,
        'acquisition_batch_size': 128,
        'pool_sample_size': 5000,
        'mc_dropout_iterations': 50,
        'tau_inv_proportion': 0.15,
        'begin_train_set_size': 75,
        'l2_penalty': 0.025,
        'save_dir': 'saved_metrics/',
        'acquisition_fn_type': 'random',
        'num_repeats': 3
    }
    filename = f'al-{configs["acquisition_fn_type"]}-{date.today()}-batch_size-{configs["acquisition_batch_size"]}'

    FORMAT = '%(asctime)s;%(levelname)s;%(message)s'
    logging.basicConfig(level=logging.DEBUG, filename= './' + filename+'.log', filemode='a', format=FORMAT)
    logging.info(configs)
    return 0

if __name__ == '__main__':
    sys.exit(main())
