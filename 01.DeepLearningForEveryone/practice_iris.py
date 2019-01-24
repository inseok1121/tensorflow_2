import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = np.loadtxt('./dataset/dataset_iris.csv', delimiter=',', dtype=np.float32)

