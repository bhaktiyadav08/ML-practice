import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/Iris.csv')
df.info()

x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df[['Species']]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=41)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy")
model.fit(x_train,y_train)

pr=model.predict(x_test)
pr

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,pr)

print(accuracy_score(y_test,pr))

print(confusion_matrix(y_test,pr))

print(classification_report(y_test,pr))

print(model.predict([[5.1,3.5,1.4,0.2]]))

from sklearn import tree
tree.plot_tree(model, filled=True)
plt.show()

