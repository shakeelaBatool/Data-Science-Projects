

# Heart Failure Prediction

### A Beginner Machine Learning Project

A beginner-level Data Science project using patient clinical data
to predict `DEATH_EVENT` using Logistic Regression.



---

## About the Project

This project uses the **Heart Failure Clinical Records Dataset**
to explore patient information and build a simple Machine Learning
model.

In this project, I used **Logistic Regression** to predict the
`DEATH_EVENT` target.

The project helped me understand the basic Machine Learning
workflow:

**Data Loading <br>→ Data Understanding <br>→ Data Processing <br>→ EDA <br>→ 
Visualization <br>→ Train-Test Split <br>→ Model Training <br>→ Evaluation**

---

## 🎯 Project Objective

The main objective of this project is to:

- Understand a real-world dataset
- Perform basic data processing
- Explore the data using EDA
- Create simple visualizations
- Train a Logistic Regression model
- Evaluate the model's performance
- Make predictions

> **Note:** This project is created for learning purposes only
> and is not intended for medical diagnosis.

---

## 📊 Dataset

The project uses the **Heart Failure Clinical Records Dataset**.

### Dataset Information

| Information | Details |
|---|---|
| Dataset | Heart Failure Clinical Records |
| Rows | 299 |
| Features | 12 |
| Target | `DEATH_EVENT` |
| Problem Type | Classification |
| Model | Logistic Regression |

### Features

- `age` — Age of the patient
- `anaemia` — Whether the patient has anaemia
- `creatinine_phosphokinase` — CPK enzyme level
- `diabetes` — Whether the patient has diabetes
- `ejection_fraction` — Percentage of blood leaving the heart
- `high_blood_pressure` — Whether the patient has high blood pressure
- `platelets` — Platelet count
- `serum_creatinine` — Serum creatinine level
- `serum_sodium` — Serum sodium level
- `sex` — Patient sex
- `smoking` — Whether the patient smokes
- `time` — Follow-up period

### Target

`DEATH_EVENT`

- `0` → Death event did not occur
- `1` → Death event occurred

---

## 📁 Project Structure

```text
Heart-Failure-Prediction/
│
├── heart_failure.csv
├── heart_failure.ipynb
└── README.md
