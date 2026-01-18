import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data={"AcidDurability":[7,7,3,1],"Strength":[7,4,4,4],"Target":['bad','bad','good','good']}
df=pd.DataFrame(data)
df

x=df[['AcidDurability','Strength']]
y=df[['Target']]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

pr=knn.predict([[3,7]])
pr
