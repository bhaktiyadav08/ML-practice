
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/USA_Housing.csv')
df.info()

xl=np.array(df.iloc[0:100,1]).reshape(-1,1)
yl=np.array(df.iloc[0:100,5]).reshape(-1,1)
yl

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xl,yl,train_size=0.8,random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
pr=model.predict(x_test)
pr

print(model.intercept_)

print(model.intercept_)

model.predict([[5.68286132]])

from sklearn import metrics
metrics.mean_absolute_error(y_test,pr)

metrics.mean_squared_error(y_test,pr)

plt.scatter(xl,yl)

plt.plot([min(xl),max(xl)],[min(pr),max(pr)],color='blue')

sns.regplot(x='Avg. Area House Age' ,y='Price',data=df)

sns.regplot(x=xl,y=yl,data=df)
