import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/USA_Housing.csv')
df.info()

x=df[['Avg. Area House Age']]
y=df[['Price']]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(criterion="squared_error",max_depth=3)
model.fit(x_train,y_train)

pr=model.predict(x_test)
pr

from sklearn import metrics
metrics.mean_absolute_error(y_test,pr)

metrics.mean_squared_error(y_test,pr)

from sklearn import tree
tree.plot_tree(model,filled=True)
plt.show()
