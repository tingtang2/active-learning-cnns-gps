# base class for active learning experiments
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import pandas as pd
import plotly.express as px
import torch

from src.data.data_loader import get_splits


class BaseTrainer(ABC):

    def __init__(self,
                 model_type,
                 optimizer_type,
                 acquisition_fn_type,
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
        self.acquisition_fn_type = acquisition_fn_type
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
        self.X_train, self.y_train, self.X_test, self.y_test = get_splits()

    @abstractmethod
    def active_train_loop(self):
        pass

    @abstractmethod
    def active_train_iteration(self):
        pass

    @abstractmethod
    def acquisition_fn(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    def save_metrics(self, metrics: List[float], iter: int):
        save_name = f'{self.acquisition_fn_type}_iteration_{iter}-batch_size-{self.acquisition_batch_size}-refactor-num-acquisitions-{self.num_acquisitions}.json'
        with open(Path(Path.home(), self.save_dir, save_name), 'w') as f:
            json.dump(metrics, f)

    @staticmethod
    def plot(name: str, metrics: List[float], save_dir: str, acquisition_fn_type: str, **kwargs) -> None:
        df = pd.DataFrame({name: [item for item in metrics], 'iteration': [i for i in range(1, len(metrics) + 1)]})
        fig = px.line(df, x='iteration', y=name, title=f'{acquisition_fn_type}: {name} as a Function of Iteration')
        fig.write_html(Path(Path.home(), save_dir, f'{acquisition_fn_type}.html'))
        fig.write_image(Path(Path.home(), save_dir, f'{acquisition_fn_type}.png'))