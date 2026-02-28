import pandas as pd

from sklearn.preprocessing import OneHotEncoder

Data={'Colour':['red','blue','green','blue','green','red'],'shape':['circle','square','circle','square','circle','square']}

df=pd.DataFrame(Data)

print(df)

df_encoded=pd.get_dummies(df)
print(df_encoded)
