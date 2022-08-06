import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, codeword_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, fc2_input_dim),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(fc2_input_dim, codeword_dim)
        )

        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(codeword_dim, fc2_input_dim),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(fc2_input_dim, 3 * 3 * 32),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # Apply linear layers
        encode = self.encoder_lin(x)
        x = self.decoder_lin(encode)
        # Unflatten
        x = self.unflatten(x)
        # Apply decode convolutions
        decode = self.decoder_conv(x)
        return decode

class Chained_AE(nn.Module):
    def __init__(self, block, num_blocks, codeword_dim, fc2_input_dim):
        super().__init__()

        self.ae = nn.ModuleList()
        for i in range(num_blocks):
            self.ae.append(self._make_layer(block, codeword_dim, fc2_input_dim))

    def _make_layer(self, block, codeword_dim, fc2_input_dim):
        layers = []
        layers.append(block(codeword_dim, fc2_input_dim))
        return nn.Sequential(*layers)

    def forward(self, x, cascade_only=False):
        if cascade_only:
            outputs = []

            for i, block in enumerate(self.ae):
                if i == 0:
                    x_next = block(x)
                else:
                    x_next = block(outputs[i-1])
                outputs.append(x_next)

            return outputs
        else:
            outputs = []
            onestep_out = []

            for i, block in enumerate(self.ae):
                if i == 0:
                    x_next = block(torch.unsqueeze(x[i],1))
                    x_1step = x_next
                else:
                    x_next = block(outputs[i-1])
                    x_1step = block(torch.unsqueeze(x[i],1))
                outputs.append(x_next)
                onestep_out.append(x_1step)

            return outputs, onestep_out
