import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv('TEPdataProc1dataCSV.csv')

X = dataset.iloc[:, 0:60].values
y = dataset.iloc[:, 60].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components=60)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(pca.components_)
with open('outputPCATrue.csv', 'w') as f:
    for item in pca.components_:
        f.write(str(item) + ", \n\n")

d = 5