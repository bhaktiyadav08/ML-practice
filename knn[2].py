import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/RidingMowers.csv')
df.info()

df.head()

x=df[['Income','Lot_Size']]
y=df[['Response']]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)

pr=knn.predict(x_test)
pr

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,pr)

