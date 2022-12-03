from cnn_gp import Sequential, Conv2d, ReLU


def initialize_base_cnn():
    model = Sequential(
        Conv2d(kernel_size=4),
        ReLU(),
        Conv2d(kernel_size=4),
        ReLU(),
        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
        Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
    )

    # self.conv1 = nn.Conv1d(in_channels=4, out_channels=self.seq_len, kernel_size=4)
    # self.pool1 = nn.MaxPool1d(kernel_size=3)
    # self.conv2 = nn.Conv1d(in_channels=self.seq_len, out_channels=self.seq_len // 2, kernel_size=4)
    # self.pool2 = nn.MaxPool1d(kernel_size=3)
    # self.dense = nn.Linear(in_features=450, out_features=MLP_out_dim)
    # self.output = nn.Linear(in_features=MLP_out_dim, out_features=output_dim)
    return model