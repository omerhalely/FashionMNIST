import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from models import Lenet5, Lenet5_Dropout, Lenet5_BN
from DataSet import DataSet
from torch.utils.data import DataLoader
import torch.optim as optim
import os


class Handler:
    def __init__(self, model, model_name, epochs, optimizer, criterion, batch_size, device, num_of_classes, weight_decay,
                 lr):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device
        self.weight_decay = weight_decay
        self.lr = lr
        self.num_of_classes = num_of_classes
        self.train_dataset = DataSet(path="./data", kind="train", num_of_classes=num_of_classes)
        self.test_dataset = DataSet(path="./data", kind="test", num_of_classes=num_of_classes)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        avg_loss = 0

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        print_every = int(len(train_dataloader) / 10)

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if batch_idx % print_every == 1 or batch_idx == len(train_dataloader) - 1:
                print(f'Epoch[{epoch:03d}] | Loss: {avg_loss:.3f}')

            self.optimizer.zero_grad()

            result = self.model(images)
            loss = self.criterion(result, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

        return avg_loss

    def evaluate_model(self, dataset, epoch):
        print("Evaluating Model")
        if self.model_name == "Lenet5_Dropout":
            self.model.eval()
        else:
            self.model.train()

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0
        correct_samples = 0
        print_every = int(len(dataloader) / 10)

        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx % print_every == 1 or batch_idx == len(dataloader) - 1:
                print(f'Epoch[{epoch:03d}] | Loss: {avg_loss:.3f} | '
                      f'Acc: {accuracy:.2f}[%] ({correct_samples} / {nof_samples})')

            with torch.no_grad():
                images = images.to(self.device)
                labels = labels.to(self.device)

                result = self.model(images)

                loss = self.criterion(result, labels)

                decision = torch.argmax(result, dim=-1)

                nof_samples += len(images)
                correct_samples += torch.sum(decision == torch.argmax(labels, dim=-1)).item()
                accuracy = 100 * (correct_samples / nof_samples)

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

        return avg_loss, accuracy

    def save_data(self, train_accuracy, test_accuracy):
        plt.plot(train_accuracy, label="Train Acc")
        plt.plot(test_accuracy, label="Test Acc")
        plt.title(f'{self.model_name} Accuracy Graph')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy[%]")
        plt.legend(loc="lower right")
        plt.xticks(np.arange(0, len(train_accuracy), 1))
        plt.grid()
        plt.savefig(os.path.join(os.getcwd(), "saved_models", self.model_name, "Accuracy Graph.png"))
        plt.close()

        train_accuracy_filepath = os.path.join(os.getcwd(), "saved_models", self.model_name, "Train Accuracy Values.log")
        test_accuracy_filepath = os.path.join(os.getcwd(), "saved_models", self.model_name, "Test Accuracy Values.log")
        train_file = open(train_accuracy_filepath, "w")
        test_file = open(test_accuracy_filepath, "w")

        for i in range(len(train_accuracy)):
            if i != len(train_accuracy) - 1:
                train_file.write(str(train_accuracy[i]) + ", ")
                test_file.write(str(test_accuracy[i]) + ", ")
            else:
                train_file.write(str(train_accuracy[i]))
                test_file.write(str(test_accuracy[i]))
        train_file.write(f'\nmax accuracy: {max(train_accuracy)}')
        test_file.write(f'\nmax accuracy: {max(test_accuracy)}')

        train_file.close()
        test_file.close()

        model_parameters_filepath = os.path.join(os.getcwd(), "saved_models", self.model_name, "Model Parameters.log")
        model_parameters_file = open(model_parameters_filepath, "w")
        model_parameters_file.write(f'model name: {self.model_name}\n')
        model_parameters_file.write(f'epochs: {self.epochs}\n')
        model_parameters_file.write(f'optimizer: {self.optimizer.__class__.__name__}\n')
        model_parameters_file.write(f'batch size: {self.batch_size}\n')
        model_parameters_file.write(f'weight decay: {self.weight_decay}\n')

        model_parameters_file.close()

    def run(self):
        print(f'Start Running {self.model_name}')
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models")):
            os.mkdir(os.path.join(os.getcwd(), "saved_models"))
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models", self.model_name)):
            os.mkdir(os.path.join(os.getcwd(), "saved_models", self.model_name))

        best_accuracy = 0
        checkpoint_filename = os.path.join(os.getcwd(), "saved_models", self.model_name, self.model_name + ".pt")

        train_accuracy_lst = []
        test_accuracy_lst = []
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}/{self.epochs}')

            self.train_one_epoch(epoch)
            train_val_loss, train_val_accuracy = self.evaluate_model(self.train_dataset, epoch)

            test_loss, test_accuracy = self.evaluate_model(self.test_dataset, epoch)

            train_accuracy_lst.append(train_val_accuracy)
            test_accuracy_lst.append(test_accuracy)

            if test_accuracy > best_accuracy:
                print(f'Saving checkpoint {checkpoint_filename}')
                state = {
                    "model": self.model.state_dict()
                }
                torch.save(state, checkpoint_filename)
                best_accuracy = test_accuracy

        self.save_data(train_accuracy_lst, test_accuracy_lst)

    def switch_model(self, model, model_name):
        self.model = model
        self.model.to(self.device)
        self.model_name = model_name
        if model_name != "Weight_Decay":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def run_all_models(self):
        print("Training all models")
        self.model_name = "Lenet5"
        self.run()

        self.switch_model(Lenet5_Dropout(output_classes=self.num_of_classes), "Lenet5_Dropout")
        self.run()

        self.switch_model(Lenet5(output_classes=self.num_of_classes), "Weight_Decay")
        self.run()

        self.switch_model(Lenet5(output_classes=self.num_of_classes), "Lenet5_BN")
        self.run()


if __name__ == "__main__":
    model_name = "Lenet5_BN"
    output_classes = 10
    epochs = 10
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_of_classes = 10
    weight_decay = 0

    model = None
    if model_name == "Lenet5":
        model = Lenet5(output_classes=output_classes)
    if model_name == "Lenet5_Dropout":
        model = Lenet5_Dropout(output_classes=output_classes)
    if model_name == "Lenet5_BN":
        model = Lenet5_BN(output_classes=output_classes)

    model.to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    handler = Handler(model=model,
                      model_name=model_name,
                      epochs=epochs,
                      optimizer=optimizer,
                      criterion=criterion,
                      batch_size=batch_size,
                      device=device,
                      num_of_classes=num_of_classes,
                      weight_decay=weight_decay,
                      lr=lr)
    handler.run()
