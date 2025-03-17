import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# loading the data from csv file to a Pandas DataFrame
data = pd.read_csv('../archive/pokemon_cleaned.csv')

