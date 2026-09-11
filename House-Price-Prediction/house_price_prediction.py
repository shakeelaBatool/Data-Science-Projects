import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('Housing.csv')
print(data.head(5))

print(f'Basic information of data\n{data.info()}')

print(data.isnull().sum())
# Removing Duplicates
data=data.drop_duplicates()
print(data)                      
# data type of the dataset
print(type(data))

# Basic statistic

print(data.describe())
# for i in data:
#     print(i)

# Data visulization

# price vs area

plt.scatter(data['area'],data['price'], color='blue')
plt.xlabel('Area of House')
plt.ylabel('Price of House')
plt.title('Price vs Area')
plt.show()

# price vs bedroom
plt.scatter(data['bedrooms'],data['price'], color='red')
plt.xlabel('Bedrooms of House')
plt.ylabel('Price of House')
plt.title('Price vs Bedrooms')
plt.show()
# correlation 
corelation=data.corr(numeric_only=True)
print(f'\n*****Corelation is*****\n{corelation}')

# visulize corelation
plt.figure(figsize=(15,10))
sns.heatmap(corelation, annot=True, cmap='cool')
plt.show()

# Data processing / ML part
data=data.replace({'yes':1,'no':0})
data = pd.get_dummies(data, drop_first=True)

print(data)
