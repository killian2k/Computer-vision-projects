import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'mlp'

        # Define the network layers in order.
        # Input is 28 * 28.
        # Output is 10 values (one per class).
        # Multiple linear layers each followed by a ReLU non-linearity (apart from the last).
        
        self.layers = nn.Sequential(
            #Part 1:
            #nn.Linear(28*28, 10)
            #[Epoch 05] Loss: 0.3091
            #[Epoch 05] Acc.: 91.6900%
            #number of parameters:
            # for each of 10 outputs there are 28*28 inputs + 1 bias
            # (28 * 28 + 1) * 10 = 7850 #parameters



            # Part 2:
            #[Epoch 04] Loss: 0.2178
            #[Epoch 04] Acc.: 93.7000%
            #Number of parameters:
            # for the hidden layer: each one of the 32 neurons has 28*28 + 1(bias) parameters 
            # it gives a subtotal of 32*(28*28 + 1) = 25120
            # for the last layer: for each one of the 10 neurons has (32 + 1(bias)) parameters
            # it gives a subtotal of 10*(32+1) = 330
            # the total of 25450

            nn.Linear(28*28,32),
            nn.ReLU(),
            nn.Linear(32,10)
        )
    
    def forward(self, batch):
        # Flatten the batch for MLP.
        b = batch.size(0)
        batch = batch.view(b, -1)
        # Process batch using the layers.
        x = self.layers(batch)
        return x


class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'conv'

        # Define the network layers in order.
        # Input is 28x28, with one channel.
        # Multiple Conv2d and MaxPool2d layers each followed by a ReLU non-linearity (apart from the last).
        # Needs to end with AdaptiveMaxPool2d(1) to reduce everything to a 1x1 image.

        '''
        1. Convolutional layer nn.Conv2d with 3 × 3 kernel and 8 channels
         followed by ReLU
         and 2 × 2 max pooling nn.MaxPool2d with stride 2.
        2. Convolutional layer nn.Conv2d with 3 × 3 kernel and 16 channels
         followed by ReLU
         and 2 × 2 max pooling nn.MaxPool2d with stride 2.
        3. Convolutional layer nn.Conv2d with 3 × 3 kernel and 32 channels
        followed by ReLU and the
        already defined adaptive average pooling
        Finally, the classifier should be a simple linear prediction layer.
        
        number of parameters
        1st layer: Conv: (1*3*3+1) * 8
        2nd layer: Conv: (8*3*3+1) * 16 
        3rd layer: Conv: (16*3*3 + 1) * 32
        Output Linear layer: 10 * (32 + 1)

        results:
        [Epoch 04] Loss: 0.0577
        [Epoch 04] Acc.: 98.4300%
        '''
        self.layers = nn.Sequential(
            Conv2d(1,8,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2),
            Conv2d(8,16,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2),
            Conv2d(16,32,(3,3)),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        # Linear classification layer.
        # Output is 10 values (one per class).
        self.classifier = nn.Sequential(
            nn.Linear(32,10)
        )
    
    def forward(self, batch):
        # Add channel dimension for conv.
        b = batch.size(0)
        batch = batch.unsqueeze(1)
        # Process batch using the layers.
        x = self.layers(batch)
        x = self.classifier(x.view(b, -1))
        return x
