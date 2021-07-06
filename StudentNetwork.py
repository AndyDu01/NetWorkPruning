import torch.nn as nn
import torch


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 50, 3),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 50, 3),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class Student():
    def __init__(self, network=StudentNet(), optimizer=torch.optim.Adam, optimizerParameter={"lr": 1e-3, "weight_decay": 1e-5}):
        self.network = network
        self.optimizer = optimizer(
            network.parameters(), **optimizerParameter)

    def save(self, PATH="."):
        Agent_Dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH="."):
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
