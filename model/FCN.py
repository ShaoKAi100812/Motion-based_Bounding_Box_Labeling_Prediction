# pytorch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions

class FCN(nn.Module):
    def __init__(self, in_channel, node_1, node_2, node_3, node_4, node_5, node_6, out_channel):
        super().__init__()
        # fully connected layers
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel, node_1),
            nn.BatchNorm1d(node_1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_1, node_2),
            nn.BatchNorm1d(node_2),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_2, node_3),
            nn.BatchNorm1d(node_3),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_3, node_4),
            nn.BatchNorm1d(node_4),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(node_4, node_5),
            nn.ReLU(),
            # nn.Dropout(p=0.3)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(node_5, node_6),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(node_6, out_channel)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        scores = self.out(x)
        return scores