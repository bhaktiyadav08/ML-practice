import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

data=np.random.normal(50,10,100)
print(data)

data.mean()

data.std()

data.min()

data.max()

data=np.append(data,[120,130,140])
print(data)

df=pd.DataFrame(data,columns=['Values'])

print(df)

sb.boxplot(x=df["Values"])
plt.title("Boxplot for outlier detection")
plt.show()

x=np.arange(len(df))

print(x)

sb.scatterplot(x=np.arange(len(df)),y=df["Values"])
plt.title("Scatterplot for outlier detection")
plt.show()

sb.scatterplot(x=df["Values"],y=np.arange(len(df)))
plt.title("Scatterplot for outlier detection")
plt.show()
