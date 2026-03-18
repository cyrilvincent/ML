import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/breast-cancer/data.csv")
y = dataframe.diagnosis
x = dataframe.drop(["id", "diagnosis"], axis=1)
np.random.seed(42)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)

scaler = pp.RobustScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

xtrain = torch.tensor(xtrain, dtype=torch.float32)
xtest = torch.tensor(xtest, dtype=torch.float32)
ytrain = torch.tensor(ytrain.values, dtype=torch.float32)
ytest = torch.tensor(ytest.values, dtype=torch.float32)

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = Net(xtrain.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training with validation
num_epochs = 100
batch_size = 16
num_batches = len(xtrain)
training_losses = []
validation_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    all_predictions = []
    all_true_labels = []

    for i in range(num_batches):
        batch_indices = torch.randperm(len(xtrain))[:batch_size]
        batch_X = xtrain[batch_indices]
        batch_y = ytrain[batch_indices]
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    training_losses.append(running_loss / num_batches)

    with torch.no_grad():
        val_outputs = model(xtest)
        val_loss = criterion(val_outputs.squeeze(), ytest)
        validation_losses.append(val_loss.item())

        val_predictions = torch.round(val_outputs).squeeze()
        all_predictions.extend(val_predictions.tolist())
        all_true_labels.extend(ytest.tolist())

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {training_losses[-1]}, Validation Loss: {validation_losses[-1]}")


