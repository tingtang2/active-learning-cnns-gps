from trainers.al_den_trainer import DenTrainer


class NpDenTrainer(DenTrainer):

    def __init__(self, oracle_save_path, **kwargs) -> None:
        super().__init__(oracle_save_path, **kwargs)