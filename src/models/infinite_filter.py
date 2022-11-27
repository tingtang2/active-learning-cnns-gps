from cnn_gp import Sequential, Conv2d, ReLU

def initialize_base_cnn():
    model = Sequential(
        Conv2d(kernel_size=3),
        ReLU(),
        Conv2d(kernel_size=3, stride=2),
        ReLU(),
        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
    )
    return model