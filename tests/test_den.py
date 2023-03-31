from models.den import Generator, DEN
from data.old_dataset import create_sequence_templates
import torch

device = torch.device('cuda')


def test_input_sequence():
    embedding_template, embedding_mask = create_sequence_templates()
    generator = Generator(embedding_template, embedding_mask, device, seq_length=101, n_classes=1).to(device)
    print(generator()[0].dtype)


def test_full_den():
    embedding_template, embedding_mask = create_sequence_templates()
    model = DEN(embedding_template, embedding_mask, device, seq_length=101, n_classes=1).to(device)

    print(model())


# test_input_sequence()
test_full_den()
