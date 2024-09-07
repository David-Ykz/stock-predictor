import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import StockPriceLSTM
from stock_info_retriever import readToDf
from preprocessor import findMovingAverage, normalizeDf, dfToTensor, splitData

import matplotlib.pyplot as plt
import numpy as np


df = readToDf("MSFT.json")
print(type(df))

