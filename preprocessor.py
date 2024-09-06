import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

def normalizeDf(df):
    scaler = MinMaxScaler()

    columns = df.columns # ignores date time column
    df[columns] = scaler.fit_transform(df[columns])
    
    return df

def dfToTensor(df, sequenceLength):
    features = df.drop(columns=['Close']).values
    target = df['Close'].values

    inputs, outputs = [], []
    for i in range(len(df) - sequenceLength):
        inputs.append(features[i:i + sequenceLength])
        outputs.append(target[i + sequenceLength])

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

def splitData(inputs, outputs, trainingPercentage):
    trainingSize = int(len(inputs) * trainingPercentage)
    inputsTrain, inputsTest = inputs[:trainingSize], inputs[trainingSize:]
    outputsTrain, outputsTest = outputs[:trainingSize], outputs[trainingSize:]
    
    return inputsTrain, inputsTest, outputsTrain, outputsTest


