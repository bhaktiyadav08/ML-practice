import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('/content/Iris.csv')
df.info()

x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['Species']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

from sklearn.svm import SVC
model=SVC(kernel='poly')
model.fit(x_train,y_train)
pr=model.predict(x_test)
pr

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(y_test,pr)

classification_report(y_test,pr)

