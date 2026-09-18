<div align="center">

# Heart Failure Prediction

### Predicting Heart Failure Outcomes Using Machine Learning

A beginner-level Machine Learning project that uses clinical
patient data and Logistic Regression to predict heart failure
death events.

</div>

---

## About the Project

Heart Failure Prediction is a supervised Machine Learning project
focused on predicting whether a death event occurred based on
clinical information about patients.

The project uses the **Heart Failure Clinical Records Dataset**
and applies **Logistic Regression** as a binary classification model.

The project follows the workflow:

**Data Loading → Data Understanding → Data Processing → EDA → 
Visualization → Train-Test Split → Logistic Regression → 
Model Evaluation → Prediction**

---

## Project Objective

The main objective of this project is to build a Machine Learning
model that can classify patients based on the `DEATH_EVENT` target.

The project helped me understand the complete beginner-level
Machine Learning workflow, from preparing clinical data to
training and evaluating a classification model.

> **Note:** This project is for educational purposes and is not
> intended for medical diagnosis or clinical decision-making.

---

## Dataset

The project uses the **Heart Failure Clinical Records Dataset**.

### Dataset Information

| Information | Details |
|---|---|
| Dataset | Heart Failure Clinical Records |
| Problem Type | Binary Classification |
| Features | 12 clinical features |
| Target | `DEATH_EVENT` |
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

## Project Structure

```text
Heart-Failure-Prediction/
│
├── heart_failure.csv
├── heart_failure.ipynb
└── README.md
