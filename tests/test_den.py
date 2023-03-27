from models.den import Generator
import torch

device = torch.device('cuda')


def test_input_sequence():
    generator = Generator(device, seq_length=109, n_classes=5).to(device)
    policy_out_1 = generator()

    print(policy_out_1.shape)

    # sequence_class, one_hots, combined = generator()
    # print(sequence_class)
    # print(one_hots)
    # print(combined.shape)


test_input_sequence()