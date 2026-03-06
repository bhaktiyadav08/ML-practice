import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/content/Iris.csv')

df

x=df[0:5]
y=df['Species']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
pr=model.predict(x_test)
pr

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,pr)

print(confusion_matrix(y_test,pr))

print(classification_report(y_test,pr))

from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,pr)
