# modules for deep exploration networks for engineering sequences
# adapted from https://github.com/johli/genesis/blob/master/analysis/splicing/definitions/generator/splirent_deconv_conv_generator_concat.py
# for generator network architecture and https://github.com/johli/genesis/blob/master/genesis/generator/genesis_generator.py
# for PWM generation and sampling procedures
import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):

    def __init__(self,
                 device: torch.device,
                 latent_dim: int = 100,
                 batch_size: int = 10,
                 seq_length: int = 101,
                 n_classes: int = 1,
                 supply_inputs: bool = False) -> None:
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        self.supply_inputs = supply_inputs
        self.device = device

        # Policy/generator network definition
        dense_1 = nn.Linear(in_features=self.latent_dim + self.n_classes, out_features=9 * 384)
        deconv_0 = nn.ConvTranspose2d(384)
        self.generator_network = nn.ModuleList([dense_1, deconv_0])

        # self.batch_norm = nn.BatchNorm1d()
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
        sequence_class = torch.randint(low=0,
                                       high=self.n_classes,
                                       size=(self.batch_size,
                                             ),
                                       dtype=torch.int32).to(self.device)

        # Get generated policy pwm logits (non-masked)
        latent_input_1 = torch.empty((self.batch_size, self.latent_dim)).to(self.device)
        latent_input_2 = torch.empty((self.batch_size, self.latent_dim)).to(self.device)

        latent_input_1.uniform_(-1.0, 1.0)
        latent_input_2.uniform_(-1.0, 1.0)

        sequence_class_onehots = torch.eye(self.n_classes).to(self.device)

        class_embedding = sequence_class_onehots.index_select(0, index=sequence_class)    # tf.gather equivalent

        seed_input_1 = torch.cat([latent_input_1, class_embedding], dim=-1)
        seed_input_2 = torch.cat([latent_input_2, class_embedding], dim=-1)

        policy_out_1 = seed_input_1
        policy_out_2 = seed_input_2
        for i, layer in enumerate(self.generator_network):
            policy_out_1 = layer(policy_out_1)
            policy_out_2 = layer(policy_out_2)

            # reshape for 2D ops
            if i == 0:
                policy_out_1.reshape(9, 1, 384)
                policy_out_2.reshape(9, 1, 384)

        return sequence_class, sequence_class_onehots, class_embedding    # , class_embedding

        # Initialize Templating and Masking Lambda layer

        # Batch Normalize PWM Logits

        # Add Template and Multiply Mask

        # Compute PWMs (Nucleotide-wise Softmax)

        # Sample proper One-hot coded sequences from PWMs

        return sequence_class


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