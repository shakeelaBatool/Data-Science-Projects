import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Load Data
data=pd.read_csv('Titanic_dataset.csv')
# print(data.head(5))

# Understand Data

print(data.info())
# print(data.shape)

# Data Cleaning
print(data.duplicated().sum())  # used to count total number of duplicate row in data

print(data["Sex"].unique())
print(data["Pclass"].unique())
print(data.isnull().sum())

# EDA

print(data["Sex"].value_counts())
print(data["Pclass"].value_counts())
print(data["Survived"].value_counts())

#Calculate survival percentage
print(data["Survived"].value_counts(normalize=True)*100)
# Check difference btw M/F survival
print(pd.crosstab(data["Sex"], data["Survived"]))

# Survival rate by gender

print(pd.crosstab(data['Sex'],data["Survived"], normalize='index')*100)
print(pd.crosstab(data["Pclass"], data["Survived"]))

print(pd.crosstab(data['Pclass'],data["Survived"], normalize='index')*100)

print(data["Age"].describe())
print(data.groupby("Survived")["Age"].mean())
print(pd.crosstab(data["SibSp"], data["Survived"]))
print(pd.crosstab(data['SibSp'],data["Survived"], normalize='index')*100)

print(data.groupby("Survived")["Age"].mean())
pd.crosstab(data["AgeGroup"], data["Survived"], normalize="index") * 100

# Visulization
import matplotlib.pyplot as plt

pd.crosstab(data["Sex"], data["Survived"]).plot(kind="bar")

plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.show()
# Statistical Analysis



# Statistical Analysis

# 1. Overall survival percentage
survival_rate = data["Survived"].value_counts(normalize=True) * 100
print("Overall Survival Rate:")
print(survival_rate)


# 2. Survival rate by Sex
sex_survival = pd.crosstab(
    data["Sex"],
    data["Survived"],
    normalize="index"
) * 100

print("\nSurvival Rate by Sex:")
print(sex_survival)


# 3. Survival rate by Pclass
class_survival = pd.crosstab(
    data["Pclass"],
    data["Survived"],
    normalize="index"
) * 100

print("\nSurvival Rate by Passenger Class:\n")
print(class_survival)


# 4. Survival rate by SibSp
sibsp_survival = pd.crosstab(
    data["SibSp"],
    data["Survived"],
    normalize="index"
) * 100

print("\nSurvival Rate by SibSp:\n")
print(sibsp_survival)


# 5. Average Age of survivors vs non-survivors
age_survival = data.groupby("Survived")["Age"].mean()

print("\nAverage Age by Survival:")
print(age_survival)

