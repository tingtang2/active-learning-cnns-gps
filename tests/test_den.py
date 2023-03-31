from models.den import Generator
from data.old_dataset import create_sequence_templates
import torch

device = torch.device('cuda')


def test_input_sequence():
    embedding_template, embedding_mask = create_sequence_templates()
    generator = Generator(device, seq_length=109, n_classes=5).to(device)
    print(generator().shape)

    # print(policy_out_1.shape)

    # sequence_class, one_hots, combined = generator()
    # print(sequence_class)
    # print(one_hots)
    # print(combined.shape)


test_input_sequence()