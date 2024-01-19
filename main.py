import torch
import torch.nn as nn
import torch.optim as optim
from models import Lenet5, Lenet5_BN, Lenet5_Dropout
from Handler import Handler
import argparse


parser = argparse.ArgumentParser(
    description="A program to build and train FashionMNIST model.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--model-name",
    type=str,
    help="The name of the model which will be trained (Lenet5/Lenet5_BN/Lenet5_Dropout/Weight_Decay/all). "
         "Default - Lenet5",
    default="all",
)

parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs the model will be trained. Default - 10",
    default=1,
)

parser.add_argument(
    "--lr",
    type=float,
    help="The learning rate of the training. Default - 1e-3",
    default=0.001,
)

parser.add_argument(
    "--batch-size",
    type=int,
    help="Size of the batches. Default - 32",
    default=32,
)

parser.add_argument(
    "--weight-decay",
    type=float,
    help="The value of weight decay which will be used for weight decay model. Default - 1e-3",
    default=0.001,
)


if __name__ == "__main__":
    args = parser.parse_args()

    model_name = args.model_name
    output_classes = 10
    epochs = args.epochs
    lr = args.lr
    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_of_classes = 10
    if model_name != "Weight_Decay":
        weight_decay = 0
    else:
        weight_decay = args.weight_decay

    print("Building Model...\n")
    model = None
    if model_name == "Lenet5" or model_name == "Weight_Decay" or model_name == "all":
        model = Lenet5(output_classes=output_classes)
    if model_name == "Lenet5_Dropout":
        model = Lenet5_Dropout(output_classes=output_classes)
    if model_name == "Lenet5_BN":
        model = Lenet5_BN(output_classes=output_classes)

    if model:
        model.to(device=device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        print(f'model name: {model_name}\nepochs: {epochs}\nlearning rate: {lr}\nbatch size: {batch_size}\n'
              f'device: {device}\nweight decay: {weight_decay}\n')
        if model_name == "all":
            handler.run_all_models()
        else:
            handler.run()

