import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import StockPriceLSTM
from stock_info_retriever import readToDf
from preprocessor import findMovingAverage, normalizeDf, dfToTensor, splitData

import matplotlib.pyplot as plt
import numpy as np

# Model parameters
INPUT_SIZE = 4   # OHLV
HIDDEN_SIZE = 64
NUM_LAYERS = 4
OUTPUT_SIZE = 5  # predict next 5 days
# LEARNING_RATE = 0.00005
LEARNING_RATE = 0.0001

# Initialize model
model = StockPriceLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)





# Process and load data
TRAINING_PERCENTAGE = 0.8
SEQUENCE_LENGTH = 5
MOVING_AVERAGE_PERIOD = 50
df = readToDf("MSFT.json")
trainDf, testDf = splitData(df, TRAINING_PERCENTAGE)
trainDf = findMovingAverage(trainDf, MOVING_AVERAGE_PERIOD)
print(trainDf)
trainDf = normalizeDf(trainDf)
testDf = normalizeDf(testDf)
trainX, trainY = dfToTensor(trainDf, SEQUENCE_LENGTH, OUTPUT_SIZE)
testX, testY = dfToTensor(testDf, SEQUENCE_LENGTH, OUTPUT_SIZE)

# Load into dataset
BATCH_SIZE = 32
trainingDataset = TensorDataset(trainX, trainY)
trainLoader = DataLoader(dataset=trainingDataset, batch_size=BATCH_SIZE, shuffle=True)
testingDataset = TensorDataset(testX, testY)
testLoader = DataLoader(dataset=testingDataset, batch_size=BATCH_SIZE)

# Training
NUM_EPOCHS = 100
lossOverTime = []
for epoch in range(NUM_EPOCHS):
    model.train()
    for batchX, batchY in trainLoader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batchX)  # outputs will have shape [batch_size, 5]
        loss = criterion(outputs, batchY)  # No need for unsqueeze, both are [batch_size, 5]
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    lossOverTime.append(loss.item())
    if (epoch+1) % 1 == 0:
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
    for i in range(len(testX)):
        batchX = testX[i]
        batchX = batchX.unsqueeze(0)
        batchY = testY[i]
        batchY = batchY.unsqueeze(0)
#        print(batchX)
        outputs = model(batchX)
        loss = criterion(outputs, batchY)  # Both outputs and batchY should have shape [batch_size, 5]
        totalLoss += loss.item() * batchX.size(0)
        numBatches += 1
    
        if numBatches == 10:
            print(batchY.numpy())

#        for i in range(len(outputs)):
        outputs = outputs.numpy()[0]
        batchY = batchY.numpy()[0]
#        print(outputs)
        if i == 0:
            predictions.extend(outputs)
            actual.extend(batchY)
        else:
            predictions.append(outputs[OUTPUT_SIZE - 1])
            actual.append(batchY[OUTPUT_SIZE - 1])

        # for prediction in batchY:
        #     actual.append(prediction[0])

        # predictions.append(outputs.numpy())
        # actual.append(batchY.numpy())



# with torch.no_grad():
#     for batchX, batchY in testLoader:
#         outputs = model(batchX)
#         loss = criterion(outputs, batchY)  # Both outputs and batchY should have shape [batch_size, 5]
#         totalLoss += loss.item() * batchX.size(0)
#         numBatches += 1
    
#         if numBatches == 10:
#             print(batchY.numpy())

#         for i in range(len(outputs)):
#             if i == 0:
#                 predictions += outputs[i]
#                 actual += batchY[i]
#             else:
#                 predictions.append(outputs[i][OUTPUT_SIZE - 1])
#                 actual.append(outputs[i][OUTPUT_SIZE - 1])

#         # for prediction in batchY:
#         #     actual.append(prediction[0])

#         # predictions.append(outputs.numpy())
#         # actual.append(batchY.numpy())

averageLoss = totalLoss / len(testX)
print(f"Average Test Loss: {averageLoss:.6f}")

# predictions = np.concatenate(predictions, axis=0)
# actual = np.concatenate(actual, axis=0)
print(len(predictions))
print(len(actual))

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Prices', color='blue', alpha=0.6)
plt.plot(predictions, label='Predicted Prices', color='red', alpha=0.6)
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title(f'Predicted vs. Actual Prices (Next {OUTPUT_SIZE} Days)')
plt.legend()
plt.grid(True)
plt.show()


# df = readToDf("AAPL.json")
# df = normalizeDf(df)
# testX, testY = dfToTensor(df, SEQUENCE_LENGTH)

# # Load into dataset
# BATCH_SIZE = 32
# testingDataset = TensorDataset(testX, testY)
# testLoader = DataLoader(dataset=testingDataset, batch_size=BATCH_SIZE)

# model.eval()
# totalLoss = 0.0
# numBatches = 0
# predictions, actual = [], []

# with torch.no_grad():
#     for batchX, batchY in testLoader:
#         batchY = batchY.unsqueeze(1)
#         outputs = model(batchX)
#         loss = criterion(outputs, batchY)
#         totalLoss += loss.item() * batchX.size(0)
#         numBatches += 1
    
#         predictions.append(outputs.numpy())
#         actual.append(batchY.numpy())

# averageLoss = totalLoss / len(testX)
# print(f"Average Test Loss: {averageLoss: .6f}")

# predictions = np.concatenate(predictions, axis=0)
# actual = np.concatenate(actual, axis=0)

# # Plot predictions vs. actual values
# plt.figure(figsize=(12, 6))
# plt.plot(actual, label='Actual Prices', color='blue', alpha=0.6)
# plt.plot(predictions, label='Predicted Prices', color='red', alpha=0.6)
# plt.xlabel('Sample Index')
# plt.ylabel('Price')
# plt.title('Predicted vs. Actual Prices')
# plt.legend()
# plt.grid(True)
# plt.show()
