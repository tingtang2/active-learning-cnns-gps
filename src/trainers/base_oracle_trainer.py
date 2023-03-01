# base class for oracle training experiments
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import pandas as pd
import plotly.express as px
import torch

from data.data_loader import get_oracle_splits, create_dataloaders


class BaseTrainer(ABC):

    def __init__(self,
                 model_type,
                 optimizer_type,
                 criterion,
                 device: str,
                 save_dir: Union[str,
                                 Path],
                 save_plots: bool = True,
                 seed: int = 11202022,
                 **kwargs) -> None:
        super().__init__()

        # basic configs every training run needs
        self.model_type = model_type
        self.optimizer_type = optimizer_type
        self.criterion = criterion
        self.device = torch.device(device)
        self.save_plots = save_plots
        self.save_dir = save_dir
        self.seed = seed

        # extra configs in form of kwargs
        for key, item in kwargs.items():
            setattr(self, key, item)

    # TODO: add additional dataset functionality
    def load_data(self) -> None:
        X_train, y_train, X_val, y_val = get_oracle_splits(seed=self.seed)
        return create_dataloaders(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, device=self.device)

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def run_experiment(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    def save_model(self, name: str):
        torch.save(self.model.state_dict(), f'{self.save_dir}models/{name}.pt')

    def save_metrics(self, metrics: List[float], name: str):
        save_name = f'{name}.json'
        with open(Path(self.save_dir, 'metrics', save_name), 'w') as f:
            json.dump(metrics, f)
