import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import StockPriceLSTM
from stock_info_retriever import readToDf
from preprocessor import normalizeDf, dfToTensor, splitData

import matplotlib.pyplot as plt
import numpy as np

# Model parameters
INPUT_SIZE = 4   # Example: 5 features (OHLCV)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1  # Predicting one value (next price)
LEARNING_RATE = 0.00005

# Initialize model
model = StockPriceLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Process and load data
TRAINING_PERCENTAGE = 0.8
df = readToDf("IBM.json")
scaledDf = normalizeDf(df)
inputTensors, outputTensors = dfToTensor(scaledDf, 5)
trainX, testX, trainY, testY = splitData(inputTensors, outputTensors, TRAINING_PERCENTAGE)

# Load into dataset
BATCH_SIZE = 32
trainingDataset = TensorDataset(trainX, trainY)
trainLoader = DataLoader(dataset=trainingDataset, batch_size=BATCH_SIZE, shuffle=True)
testingDataset = TensorDataset(testX, testY)
testLoader = DataLoader(dataset=testingDataset, batch_size=BATCH_SIZE)

# Training
NUM_EPOCHS = 5000
lossOverTime = []
for epoch in range(NUM_EPOCHS):
    model.train()
    for batchX, batchY in trainLoader:
        optimizer.zero_grad()
        batchY = batchY.unsqueeze(1)  # Adds an extra dimension to make batchY [batch_size, 1]
    
        # Forward pass
        outputs = model(batchX)
        loss = criterion(outputs, batchY)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    lossOverTime.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')

print(len(lossOverTime))
# Plotting training loss
plt.figure(figsize=(10, 5))
plt.plot(range(NUM_EPOCHS), lossOverTime, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()




model.eval()
totalLoss = 0.0
numBatches = 0
predictions, actual = [], []

with torch.no_grad():
    for batchX, batchY in testLoader:
        batchY = batchY.unsqueeze(1)
        outputs = model(batchX)
        loss = criterion(outputs, batchY)
        totalLoss += loss.item() * batchX.size(0)
        numBatches += 1
    
        predictions.append(outputs.numpy())
        actual.append(batchY.numpy())

averageLoss = totalLoss / len(testX)
print(f"Average Test Loss: {averageLoss: .6f}")

predictions = np.concatenate(predictions, axis=0)
actual = np.concatenate(actual, axis=0)

# Plot predictions vs. actual values
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Prices', color='blue', alpha=0.6)
plt.plot(predictions, label='Predicted Prices', color='red', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Predicted vs. Actual Prices')
plt.legend()
plt.grid(True)
plt.show()
