from stock_info_retriever import readToDf
from preprocessor import normalizeDf, dfToTensor, splitData

df = readToDf("IBM.json")
scaledDf = normalizeDf(df)

inputTensors, outputTensors = dfToTensor(scaledDf, 5)

print(splitData(inputTensors, outputTensors, 0.8))