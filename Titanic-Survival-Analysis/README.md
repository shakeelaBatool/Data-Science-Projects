<div align="center">

# Titanic Survival Analysis

### Exploring Survival Patterns Through Data

A beginner-level Data Science project focused on exploring,
cleaning, analyzing, and visualizing the famous Titanic dataset.



---

##  About the Project  </div>


The Titanic Survival Analysis project explores passenger data from the
Titanic disaster to understand the factors associated with survival.

Using Python and Pandas, I performed data cleaning, exploratory data
analysis (EDA), statistical analysis, and visualization to identify
patterns in passenger survival.

The analysis mainly focuses on:

- Gender
- Passenger Class
- Age
- Family-related information
- Overall Survival

---

## Project Objective

The main objective of this project is to understand how different
passenger characteristics were associated with survival.

This project helped me build a practical foundation in:

**Data Understanding → Data Cleaning → EDA → Statistical Analysis → Visualization → Insights**

---

## Dataset

The project uses the **Titanic passenger dataset**.

| Information | Details |
|---|---|
| Dataset | Titanic Dataset |
| Samples | 891 passengers |
| Target | `Survived` |
| Target Values | `0 = Did not survive`, `1 = Survived` |
| Main Features | `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, etc. |

### Main Features

- `Pclass` — Passenger class
- `Name` — Passenger name
- `Sex` — Passenger gender
- `Age` — Passenger age
- `SibSp` — Number of siblings/spouses aboard
- `Parch` — Number of parents/children aboard
- `Fare` — Ticket fare
- `Cabin` — Cabin information
- `Embarked` — Port of embarkation
- `Survived` — Survival status

---

## Project Structure

Titanic-Survival-Analysis/
│
├── Titanic_dataset.csv
├── Titanic_Survival_Analysis.ipynb
└── README.md
## 📓 Notebook

This project contains one Jupyter Notebook:

| Notebook | Description |
|---|---|
| `Titanic_Survival_Analysis.ipynb` | Complete data analysis including data understanding, cleaning checks, EDA, statistical analysis, visualization, and insights |

The notebook follows a beginner-friendly Data Science workflow from
loading the dataset to extracting meaningful insights.
---
---

## Analysis Performed

### 1. Data Understanding

The dataset was explored using:

- `info()`
- `shape`
- `columns`
- `describe()`
- `unique()`

This helped understand the dataset structure, data types,
statistical information, and categorical values.

### 2. Data Cleaning

The following checks were performed:

- Missing value analysis
- Duplicate row detection
- Unique value analysis
- Data consistency checks

### 3. Exploratory Data Analysis

The project explores:

- Gender distribution
- Passenger class distribution
- Survival distribution
- Age distribution
- SibSp distribution
- Survival patterns across different groups

### 4. Statistical Analysis

The following Pandas functions were used:

- `value_counts()`
- `value_counts(normalize=True)`
- `groupby()`
- `pd.crosstab()`
- `describe()`

These were used to calculate and compare survival patterns
across different passenger groups.

---

## Results of the Project

### Overall Survival

| Survival Status | Passengers | Percentage |
|---|---:|---:|
| ✅ Survived | 342 | **38.4%** |
| ❌ Did Not Survive | 549 | **61.6%** |

### Survival by Gender

| Gender | Did Not Survive | Survived | Survival Rate |
|---|---:|---:|---:|
| Female | 81 | 233 | **74.2%** |
| Male | 468 | 109 | **18.9%** |

### Survival by Passenger Class

| Passenger Class | Did Not Survive | Survived | Survival Rate |
|---|---:|---:|---:|
| 1st Class | 80 | 136 | **63.0%** |
| 2nd Class | 97 | 87 | **47.3%** |
| 3rd Class | 372 | 119 | **24.2%** |

---

## Key Findings

### Gender

Gender showed a significant difference in survival rates.

- Female survival rate: **74.2%**
- Male survival rate: **18.9%**

Female passengers in this dataset had a much higher survival rate
than male passengers.

### Passenger Class

Passenger class was also strongly associated with survival.

- 1st Class: **63.0%**
- 2nd Class: **47.3%**
- 3rd Class: **24.2%**

Passengers in higher classes had higher survival rates than those
in lower classes.

### Age

Age was analyzed by comparing the average age of survivors and
non-survivors and by examining different age groups.

### SibSp

The number of siblings or spouses aboard was analyzed to explore
differences in survival rates among different groups.

### Overall Survival

Out of 891 passengers:

- **342 passengers survived**
- **549 passengers did not survive**
- Overall survival rate was approximately **38.4%**

---

## Data Visualization

Visualizations were created to make the analysis and patterns
easier to understand.

The project includes visual analysis of:

- Survival by gender
- Passenger class
- Survival distribution
- Age-related patterns
- Other categorical relationships

### Example: Survival by Gender

```python
pd.crosstab(
    data["Sex"],
    data["Survived"]
).plot(kind="bar")

plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)

plt.show()
