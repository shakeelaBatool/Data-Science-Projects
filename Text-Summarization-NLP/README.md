# Interactive Text Summarization using NLP

## 📌 Project Overview

This project focuses on **automatic text summarization using Natural Language Processing (NLP)**.

The system takes a long customer review and extracts the most important sentence as a short summary. The project uses a **TF-IDF-based extractive summarization approach** and evaluates the generated summaries using **ROUGE metrics**.

An interactive **Streamlit dashboard** is also included, allowing users to enter their own customer reviews and generate summaries.

---

## 🎯 Problem Statement

Online platforms contain a large number of customer reviews. Reading long reviews can be time-consuming, especially when users only need the main information.

This project aims to automatically extract the most important sentence from a customer review and present it as a short summary.

---

## 💡 Objective

The main objectives of this project are:

* Understand basic Natural Language Processing concepts.
* Preprocess customer review text.
* Split reviews into individual sentences.
* Calculate sentence importance using TF-IDF.
* Extract the most important sentence as a summary.
* Evaluate the generated summaries using ROUGE.
* Compare different TF-IDF summarization approaches.
* Build an interactive Streamlit dashboard.

---

## 📊 Dataset

The project uses the **Amazon Fine Food Reviews** dataset.

The dataset contains customer reviews of food products.

### Important Columns

| Column      | Description                 |
| ----------- | --------------------------- |
| `Text`      | Full customer review        |
| `Summary`   | Human-written short summary |
| `Score`     | Customer rating             |
| `ProductId` | Product identifier          |
| `UserId`    | User identifier             |
| `Time`      | Review timestamp            |

The original dataset contains **568,454 reviews**.

For this project:

* `Text` is used as the input review.
* `Summary` is used as the reference summary for evaluation.

Rows with missing values in the `Summary` column were removed because the reference summary is required for ROUGE evaluation.

---

## 🧹 Data Preprocessing

The preprocessing in this project is kept simple because the focus is on understanding NLP-based summarization.

### Steps

1. Load the dataset using Pandas.
2. Remove rows with missing reference summaries.
3. Convert review text to lowercase.
4. Split the review into sentences using NLTK sentence tokenization.

---

## 🧠 NLP Methodology

The project uses **extractive text summarization**.

Extractive summarization selects important sentences directly from the original text instead of generating completely new sentences.

### Workflow

```text
Customer Review
       ↓
Text Preprocessing
       ↓
Sentence Tokenization
       ↓
TF-IDF
       ↓
Sentence Scoring
       ↓
Select Highest-Scoring Sentence
       ↓
Generated Summary
```

---

## ⚙️ How the Model Works

### 1. Sentence Tokenization

The review is divided into individual sentences using NLTK.

For example:

```text
The product is good. The quality is excellent. I will buy it again.
```

becomes:

```text
Sentence 1
Sentence 2
Sentence 3
```

### 2. TF-IDF

TF-IDF (**Term Frequency-Inverse Document Frequency**) is used to calculate the importance of words within the sentences.

Important words receive higher TF-IDF values.

### 3. Sentence Scoring

The TF-IDF values of the words in each sentence are summed to calculate a sentence score.

```text
Sentence Score = Sum of TF-IDF values
```

### 4. Sentence Selection

The sentence with the highest score is selected as the generated summary.

The final model uses:

**TF-IDF Sum + Top 1 Sentence**

---

## 📈 Model Evaluation

The generated summaries were evaluated using three ROUGE metrics:

* **ROUGE-1** — measures unigram overlap.
* **ROUGE-2** — measures bigram overlap.
* **ROUGE-L** — measures longest common subsequence overlap.

The experiments were conducted on **100 reviews**.

### Experimental Results

| Experiment          |    ROUGE-1 |    ROUGE-2 |    ROUGE-L |
| ------------------- | ---------: | ---------: | ---------: |
| TF-IDF Sum + Top 2  |     0.0996 |     0.0311 |     0.0903 |
| TF-IDF Sum + Top 1  | **0.1168** | **0.0416** | **0.1056** |
| TF-IDF Mean + Top 2 |     0.0996 |     0.0311 |     0.0903 |

Based on these experiments, **TF-IDF Sum + Top 1** produced the highest ROUGE values among the tested approaches and was selected as the final method.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Users can:

* Enter a customer review.
* Generate a summary.
* View the generated summary.
* View the number of original sentences.
* View sentence importance scores.
* Identify which sentence was selected by the model.

### Dashboard Workflow

```text
Enter Review
     ↓
Click "Generate Summary"
     ↓
NLP Processing
     ↓
TF-IDF Sentence Scoring
     ↓
Top 1 Sentence
     ↓
Generated Summary
```

### Dashboard Screenshot

Add your dashboard screenshot here:

```text
![Streamlit Dashboard](screenshots/dashboard.png)
```

---

## 🧪 Testing

The dashboard was tested using different types of customer reviews, including:

* Positive reviews
* Negative reviews
* Mixed-feedback reviews
* Longer reviews
* Reviews containing multiple opinions

The purpose of testing was to observe whether the selected sentence represents the main idea of the original review.

---

## 🔍 Key Findings

* TF-IDF can be used to identify important sentences in customer reviews.
* Extractive summarization selects sentences directly from the original review.
* In the tested experiments, selecting one sentence produced higher ROUGE values than selecting two sentences.
* Human-written summaries can be much shorter than extracted summaries.
* ROUGE is based on word overlap, so it may not completely represent semantic similarity.
* The final model is simple and suitable for understanding the basic workflow of NLP-based summarization.

---

## ⚠️ Limitations

* The system is **extractive**, so it cannot generate new sentences.
* The selected sentence may not always represent the complete meaning of a long review.
* Human-written summaries are often very short, while extracted sentences can be longer.
* ROUGE measures lexical overlap and does not completely evaluate meaning.
* Model evaluation was performed on a sample of 100 reviews.

---

## 🚀 Future Improvements

Possible future improvements include:

* Abstractive text summarization.
* Transformer-based summarization models.
* Better control over summary length.
* Evaluation on a larger sample of reviews.
* Comparison with advanced NLP summarization techniques.
* Improved dashboard design and deployment.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **NLTK**
* **Scikit-learn**
* **ROUGE**
* **Streamlit**
* **Matplotlib**
* **Seaborn**

---

## 📂 Project Structure

```text
Text-Summarization-NLP/
│
├── app.py
├── Text-Summarization-NLP.ipynb
├── README.md
└── screenshots/
    └── dashboard.png
```

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/shakeelaBatool/Data-Science-Projects.git
```

### 2. Navigate to the Project Folder

```bash
cd Data-Science-Projects/Text-Summarization-NLP
```

### 3. Install Required Libraries

```bash
pip install pandas numpy nltk scikit-learn rouge-score streamlit matplotlib seaborn
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your web browser.

---

## 👩‍💻 Author

**Shakeela Batool**

BS Mathematics Student
Namal University

Interested in:

* Data Science
* Machine Learning
* Natural Language Processing
* Artificial Intelligence

---

## 📌 Project Status

**Completed — Beginner-Level NLP Project**

The project includes:

* Dataset Analysis ✅
* Text Preprocessing ✅
* Sentence Tokenization ✅
* TF-IDF Summarization ✅
* ROUGE Evaluation ✅
* Model Experiments ✅
* Final Model ✅
* Streamlit Dashboard ✅
* Testing ✅
* GitHub Documentation ✅
