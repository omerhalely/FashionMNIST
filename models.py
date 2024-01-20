import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class Lenet5(nn.Module):
    def __init__(self, output_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.fc1 = nn.Linear(int(16 * (28 / 4) * (28 / 4)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Lenet5_Dropout(nn.Module):
    def __init__(self, output_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.fc1 = nn.Linear(int(16 * (28 / 4) * (28 / 4)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(84, output_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Lenet5_BN(nn.Module):
    def __init__(self, output_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(int(16 * (28 / 4) * (28 / 4)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = Lenet5(output_classes=10)

    image = torch.rand(1, 1, 28, 28)

    output = model(image)

    print(output.shape)
    print(output)
