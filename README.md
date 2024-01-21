# FashionMNIST
FashionMNIST image classification with Lenet5.

We implemented 3 classification models:
1. Lenet5
2. Lenet5_Dropout
3. Lenet5_BN

## Dataset
We use the FashionMNIST dataset arranged according to the following:
```
data:
    |   test-images-idx3-ubyte.gz
    |   test-labels-idx1-ubyte.gz
    |   train-images-idx3-ubyte.gz
    |   train-labels-idx1-ubyte.gz
```
## Training
For training a specific model:
```bash
python main.py --model-name "model-name" --epochs 20 --batch-size 64
```
The model will be trained and saved at a directory named ./models/model_name which will be created automatically.

Possible values for model-name are: Lenet5, Lenet5_Dropout, Lenet5_BN, Weight_Decay.
When training Lenet5, Lenet5_Dropout, Lenet5_BN, the value of weight decay is automatically set to 0.

Example for training Lenet5:
```bash
python main.py --model-name "Lenet5" --epochs 20 --batch-size 64
```
When training the model with weight decay, the default value is 0.001.
For using a different value use the weight-decay flag:
```bash
python main.py --model-name "Weight_Decay" --epochs 20 --batch-size 64 --weight-decay 0.002
```
For training all the models, you can use the value "all" for model-name:
```bash
python main.py --model-name "all" --epochs 20 --batch-size 64 --weight-decay 0.001
```
This line will train all the models and will save each model at ./models/model_name.
The entered value for weight decay will be used only for the case when we train Lenet5 with weight decay.
For all the other models, the value of weight decay will be set to 0.

## Testing
For testing a specific model:
```bash
python main.py --model-name "model-name" --test "dataset to test on (test/train)"
```
For example, in order to test Lenet5 on the test data:
```bash
python main.py --model-name "Lenet5" --test "test"
```
If there is a pretrained model placed in ./models/Lenet5, the model will be loaded and tested
over the test data. Otherwise, we use an untrained model.
