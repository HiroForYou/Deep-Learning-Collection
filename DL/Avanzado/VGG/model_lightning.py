import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512,'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,'M']
}
# Luego se aplanan y se pasan a la red densa de 3 capas

class VGG_net(pl.LightningModule):

    def __init__(self, in_channels=3, num_classes=1000, lr=1e-3, linear_units=100):
        super().__init__()
        self.in_channels = in_channels
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.conv_layers = self.create_conv_layer(VGG_types['VGG16'])
        self.fcs = nn.Sequential(
            #nn.Linear(512*7*7, linear_units), # 7 = 224/(2**5)  5 max pools
            nn.Linear(512*1*1, linear_units), # 1 = 32/(2**5)  5 max pools
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(linear_units, linear_units),
            #nn.Linear(4096, 4096), # disminuir las unidades a de 4096 a 100
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(linear_units, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels

        for item in architecture:
            if type(item) == int:
                out_channels = item

                layers += [
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  
                            nn.BatchNorm2d(item),
                            nn.ReLU()
                          ]

                in_channels = item
            elif item == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)
    
    def loss(self, y_hat, y):
        l = self.criterion(y_hat, y)
        return l

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y.long())

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG_net(in_channels=3, num_classes=10, lr=1e-3, linear_units=10).to(device)
    x = torch.randn(1, 3, 64, 64).to(device)
    y_hat = model(x)
    y = torch.empty(1, dtype=torch.long).random_(5).to(device)
    print(y_hat)











