import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderWithClassifier(nn.Module):
    def __init__(self, autoencoder, dim_z, class_nums):
        super(AutoencoderWithClassifier, self).__init__()
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.classifier = nn.Sequential(
            nn.Linear(dim_z, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, class_nums)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.classifier(encoded.squeeze().detach())
        decoded = self.decoder(encoded)
        return encoded,output,decoded

 
