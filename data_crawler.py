import requests
import sklearn as sk
import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

df = pd.read_csv(url,header = None,names = ['Sepal_Length','Sepal_width','Petal_Length','Petal_Width','Labels'])

df.to_csv('dataset.csv')