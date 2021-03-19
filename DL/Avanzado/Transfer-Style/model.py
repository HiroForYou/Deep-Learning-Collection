import torch.nn as nn
from torchvision import models


class VGG_styler(nn.Module):
    
    def __init__(self):
        super(VGG_styler, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = [] 

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features        
