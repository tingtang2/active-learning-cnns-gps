# modules for deep exploration networks for engineering sequences
import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):

    def __init__(self) -> None:
        super(Generator, self).__init__()

        self.batch_norm = nn.BatchNorm1d()
        self.batch_size = 10

        # generator_model = Model(inputs=[sequence_class_input] + generator_inputs,
        #                         outputs=[
        #                             sequence_class,
        #                             pwm_logits_1,
        #                             pwm_logits_2,
        #                             pwm_1,
        #                             pwm_2,
        #                             sampled_pwm_1,
        #                             sampled_pwm_2,
        #                             onehot_mask,
        #                             sampled_onehot_mask
        #                         ] + extra_outputs)

    def forward(self, random_seed: int = None):
        # Seed class input for all dense/embedding layers
        sequence_class_input = torch.ones(self.batch_size)

        # Get generated policy pwm logits (non-masked)

        # Initialize Templating and Masking Lambda layer

        # Batch Normalize PWM Logits

        # Add Template and Multiply Mask

        # Compute PWMs (Nucleotide-wise Softmax)

        # Sample proper One-hot coded sequences from PWMs

        pass


class Predictor(nn.Module):

    def __init__(self) -> None:
        super(Predictor, self).__init__()

    def forward():
        pass


# nn.Module class for Deep Exploration Network
class DEN(nn.Module):

    def __init__(self) -> None:
        super(DEN, self).__init__()

    def forward():
        pass