# modules for deep exploration networks for engineering sequences
# adapted from https://github.com/johli/genesis/blob/master/analysis/splicing/definitions/generator/splirent_deconv_conv_generator_concat.py
# for generator network architecture and https://github.com/johli/genesis/blob/master/genesis/generator/genesis_generator.py
# for PWM generation and sampling procedures
import torch
import torch.nn.functional as F
from torch import nn

from models.base_modules import StraightThroughEstimator


class GeneratorNetwork(nn.Module):

    def __init__(self, latent_dim: int = 100, batch_size: int = 32, seq_length: int = 101, n_classes: int = 1) -> None:
        super(GeneratorNetwork, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        # Policy/generator network definition
        dense_0 = nn.Linear(in_features=self.latent_dim + self.n_classes, out_features=9 * 384)

        deconv_0 = nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=(7, 1), stride=(2, 1))
        batch_norm_0 = nn.BatchNorm2d(num_features=256)

        deconv_1 = nn.ConvTranspose2d(in_channels=256, out_channels=192, kernel_size=(8, 1), stride=(2, 1))
        batch_norm_1 = nn.BatchNorm2d(num_features=192)

        deconv_2 = nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=(7, 1), stride=(2, 1))
        batch_norm_2 = nn.BatchNorm2d(num_features=128)

        conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(8, 1), stride=(1, 1), padding='same')
        batch_norm_3 = nn.BatchNorm2d(num_features=128)

        conv_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(8, 1), stride=(1, 1), padding='same')
        batch_norm_4 = nn.BatchNorm2d(num_features=64)

        conv_5 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=(8, 1), stride=(1, 1), padding='same')

        self.generator_network = nn.ModuleList([
            dense_0,
            deconv_0,
            batch_norm_0,
            deconv_1,
            batch_norm_1,
            deconv_2,
            batch_norm_2,
            conv_3,
            batch_norm_3,
            conv_4,
            batch_norm_4,
            conv_5,
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.generator_network):
            x = layer(x)

            # reshape for 2D ops
            if i == 0:
                x = x.reshape(self.batch_size, 384, 9, 1)

            # if i == 7:
            #     break

            if isinstance(layer, nn.BatchNorm2d):
                x = F.relu(x)

        return x.reshape(self.batch_size, self.seq_length, 4, 1)


class Generator(nn.Module):

    def __init__(self,
                 device: torch.device,
                 latent_dim: int = 100,
                 batch_size: int = 32,
                 seq_length: int = 101,
                 n_classes: int = 1,
                 n_samples: int = 10,
                 supply_inputs: bool = False) -> None:
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.n_samples = n_samples

        self.supply_inputs = supply_inputs
        self.device = device

        self.generator_network = GeneratorNetwork(latent_dim=latent_dim,
                                                  batch_size=batch_size,
                                                  seq_length=seq_length,
                                                  n_classes=n_classes).to(self.device)

        self.onehot_template_layer = nn.Embedding(n_classes, seq_length * 4)
        nn.init.constant_(self.onehot_template_layer.weight, 0)

        self.onehot_mask_layer = nn.Embedding(n_classes, seq_length * 4)
        nn.init.constant_(self.onehot_mask_layer.weight, 1)

        self.straight_through = StraightThroughEstimator()

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

        raw_logits_1 = self.generator_network(seed_input_1)
        raw_logits_2 = self.generator_network(seed_input_2)

        onehot_template = self.onehot_template_layer(sequence_class).reshape(self.batch_size, self.seq_length, 4, 1)
        onehot_mask = self.onehot_mask_layer(sequence_class).reshape(self.batch_size, self.seq_length, 4, 1)

        # Add Template and Multiply Mask
        pwm_logits_1 = raw_logits_1 * onehot_mask + onehot_template
        pwm_logits_2 = raw_logits_2 * onehot_mask + onehot_template

        # Compute PWMs (Nucleotide-wise Softmax)
        pwm_1 = F.softmax(pwm_logits_1, dim=-2)
        pwm_2 = F.softmax(pwm_logits_2, dim=-2)

        # Sample proper One-hot coded sequences from PWMs
        # Optionally tile each PWM to sample from and create sample axis

        pwm_logits_upsampled_1 = torch.tile(pwm_logits_1, (self.n_samples, 1, 1, 1))
        pwm_logits_upsampled_2 = torch.tile(pwm_logits_2, (self.n_samples, 1, 1, 1))
        sampled_onehot_mask = torch.tile(onehot_mask,
                                         (self.n_samples,
                                          1,
                                          1,
                                          1)).reshape(self.batch_size,
                                                      self.n_samples,
                                                      self.seq_length,
                                                      4,
                                                      1)

        sampled_pwm_1 = self.sample_pwm(pwm_logits_upsampled_1)

        sampled_pwm_2 = self.sample_pwm(pwm_logits_upsampled_2)

        # sampled_onehot_mask = Lambda(lambda x: K.permute_dimensions(K.reshape(x, (n_samples, batch_size, seq_length, 4, 1)), (1, 0, 2, 3, 4)))(sampled_onehot_mask)

        # Lock all generator layers except policy layers
        return sampled_pwm_1, sampled_pwm_2, sampled_pwm_1

    def sample_pwm(self, pwm_logits: torch.Tensor) -> torch.Tensor:
        flat_pwm = pwm_logits.reshape(-1, 4)
        nt_probs = F.softmax(flat_pwm, dim=-1)

        # implement straight-through gradient estimation
        if self.training:
            sampled_onehot = F.one_hot(torch.multinomial(input=nt_probs, num_samples=1).squeeze(1))
        else:
            sampled_onehot = F.one_hot(nt_probs.argmax(1))

        sampled_pwm = self.straight_through(sampled_onehot, nt_probs)

        return sampled_pwm.reshape(self.batch_size, self.n_samples, self.seq_length, 4, 1)


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