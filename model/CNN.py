import torch.nn as nn

def flatten(x):
    N = x.shape[0]          # read in N, C, H, W
    return x.view(N, -1)    # "flatten" the C * H * W values into a single vector per image

class CNN(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3, channel_4, channel_5, node_1, node_2, node_3, node_4, out_channel):
        super().__init__()
        # convolution layers
        # input image size = sample_size * 1 * 84 * 112
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channel, channel_1, (7,7), padding=3, stride=1),
            nn.BatchNorm2d(channel_1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )   # output_size = sample_size * channel_1 * 42 * 56 
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_1, channel_2, (7,7), padding=3, stride=1),
            nn.BatchNorm2d(channel_2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )   # output_size = sample_size * channel_2 * 21 * 28
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_2, channel_3, (7,7), padding=3, stride=1),
            nn.BatchNorm2d(channel_3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )   # output_size = sample_size * channel_3 * 10 * 14
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_3, channel_4, (7,7), padding=3, stride=1),
            nn.BatchNorm2d(channel_4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )   # output_size = sample_size * channel_4 * 5 * 7
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel_4, channel_5, (3,3), padding=1, stride=1),
            nn.BatchNorm2d(channel_5),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
        )   # output_size = sample_size * channel_5 * 2 * 3
        # fully connected layers
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_5*5*7, node_1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_1, node_2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_2, node_3),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_3, node_4),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(node_4, out_channel)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        scores = self.out(x)
        return scores