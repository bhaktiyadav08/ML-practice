import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df2=pd.read_csv('/content/data.csv')

df2.info()

x=df2[['R&D Spend','Administration','Marketing Spend']]
y=df2['Profit']
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
pr=model.predict(x_test)
pr

print(model.intercept_)

print(model.coef_)

model.predict([[165349.20,136897.80,471784.10]])

from sklearn import metrics
metrics.mean_absolute_error(y_test,pr)

metrics.mean_squared_error(y_test,pr)
