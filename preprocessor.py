import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

def findMovingAverage(df, timePeriod):
    print(type(df))
    newDf = df.copy()
    newDf['Open'] = df['Open'].rolling(window=timePeriod).mean()
    newDf['Close'] = df['Close'].rolling(window=timePeriod).mean()
    newDf['High'] = df['High'].rolling(window=timePeriod).mean()
    newDf['Low'] = df['Low'].rolling(window=timePeriod).mean()
    newDf['Volume'] = df['Volume'].rolling(window=timePeriod).mean()
    
    newDf.dropna(inplace=True)

    return newDf

def normalizeDf(df):
    scaler = MinMaxScaler()

    columns = df.columns # ignores date time column
    df[columns] = scaler.fit_transform(df[columns])
    
    return df

def splitData(df, trainingPercentage):
    trainingSize = int(len(df) * trainingPercentage)
    trainDf = df[:trainingSize]
    testDf = df[trainingSize:]

    return trainDf, testDf

def dfToTensor(df, sequenceLength, outputWindow):
    features = df.drop(columns=['Close']).values
    target = df['Close'].values

    inputs, outputs = [], []
    for i in range(len(df) - sequenceLength - outputWindow + 1):
        inputs.append(features[i:i + sequenceLength])
        outputs.append(target[i + sequenceLength:i + sequenceLength + outputWindow])

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32)

# def splitData(inputs, outputs, trainingPercentage):
#     trainingSize = int(len(inputs) * trainingPercentage)
#     inputsTrain, inputsTest = inputs[:trainingSize], inputs[trainingSize:]
#     outputsTrain, outputsTest = outputs[:trainingSize], outputs[trainingSize:]
    
#     return inputsTrain, inputsTest, outputsTrain, outputsTest


