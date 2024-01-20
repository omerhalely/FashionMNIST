import torch
from torch.utils.data import Dataset
from utils import load_mnist
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import transforms


class DataSet(Dataset):
    def __init__(self, path, kind, num_of_classes):
        images, labels = load_mnist(path=path, kind=kind)
        self.images = images
        self.labels = labels
        self.num_of_classes = num_of_classes
        self.process_data()

    def process_data(self):
        self.images = torch.from_numpy(np.copy(self.images))
        self.images = torch.reshape(self.images, (self.images.shape[0], 1, 28, 28))
        self.labels = torch.from_numpy(np.copy(self.labels))
        self.labels = torch.reshape(self.labels, (self.labels.shape[0], 1))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx].float()
        # mean = torch.mean(image)
        # std = torch.std(image)
        mean = 0.1307
        std = 0.3081
        transform = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])
        label = self.labels[idx]
        return_label = torch.zeros(self.num_of_classes)
        return_label[int(label)] = 1
        return transform(image), return_label.float()


if __name__ == "__main__":
    path = "./data"
    kind = "train"
    num_of_classes = 10

    dataset = DataSet(path=path, kind=kind, num_of_classes=num_of_classes)
    print("Dataset Length:", len(dataset))

    labels = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    rows = 2
    cols = 3
    fig, axis = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            random_idx = int(random.random() * len(dataset))
            image, label = dataset[random_idx]
            label = torch.argmax(label)
            axis[i, j].imshow(torch.permute(image, (1, 2, 0)), cmap="gray")
            axis[i, j].set_title(labels[int(label)])

    plt.show()

