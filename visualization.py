import matplotlib.pyplot as plt
import pandas as pd

x=[10,20,30,40]
y=[20,25,35,55]
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("Line chart")
plt.plot(x,y)

data=pd.read_csv('tips.csv')
x=data['day']
y=data['total_bill']
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("Bar chart")
plt.bar(x,y)

cars=['Audi','BMW','Ford','Tesla','Jaguar']
data=[23,10,35,15,12]
plt.title("Car data")
plt.pie(data,labels=cars)
plt.show()

data=pd.read_csv('tips.csv')
x=data['day']
y=data['total_bill']
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("Scattered chart")
plt.scatter(x,y)
plt.show()

data=pd.read_csv('tips.csv')
x=data['day']
y=data['total_bill']
plt.xlabel('Total bill')
plt.ylabel('Frequency')
plt.title("Histogram")
plt.hist(y)
plt.show()
