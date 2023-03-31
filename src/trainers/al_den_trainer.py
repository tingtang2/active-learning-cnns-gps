import logging
import math
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from data.data_loader import create_test_dataloader
from data.old_dataset import create_sequence_templates
from models.den import DEN
from trainers.base_trainer import BaseTrainer


class AlDenTrainer(BaseTrainer):

    def __init__(self, device, **kwargs) -> None:
        super().__init__(**kwargs)

        embedding_template, embedding_mask = create_sequence_templates()

        self.den = DEN(embedding_template=embedding_template, embedding_mask=embedding_mask, device=device)
