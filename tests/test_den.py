from models.den import Generator
import torch

device = torch.device('cuda')


def test_input_sequence():
    generator = Generator(device, n_classes=2)

    sequence_class, one_hots, combined = generator()
    print(sequence_class)
    print(one_hots)
    print(combined.shape)


test_input_sequence()