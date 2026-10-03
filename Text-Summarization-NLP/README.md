# Text Summarization using NLP

## 1. Text Summarization

Text summarization is an **Natural Language Processing (NLP)** task that automatically converts a long piece of text into a shorter version while preserving its important information.

In this project, **extractive text summarization** is used. Instead of generating new sentences, the system identifies the most important sentence from the original customer review and uses it as the summary.

The project uses **TF-IDF and sentence position** to calculate the importance of each sentence.

### How It Works

```text
Customer Review
       ↓
Sentence Tokenization
       ↓
TF-IDF Score
       +
Position Score
       ↓
Combined Sentence Score
       ↓
Highest-Scoring Sentence
       ↓
Generated Summary
```

---

## 2. Problem Statement

Online platforms contain a large number of customer reviews. Many reviews are long and contain multiple sentences, making them time-consuming to read.

The objective of this project is to develop an NLP-based system that can automatically identify the most important sentence from a customer review and present it as a short summary.

The project focuses on **extractive summarization**, where important sentences are selected directly from the original review.

---

## 3. Dataset

### Dataset Used

This project uses the **Amazon Fine Food Reviews Dataset**.

The dataset contains customer reviews of food products purchased on Amazon. Each record contains information about the reviewer, product, rating, review text, and a human-written summary.

The main file used in this project is:

```text
Reviews.csv
```

### Dataset Size

The original dataset contains:

* **568,454 reviews**
* **10 columns**

After removing records with missing values in the `Summary` column, the remaining data is used for summarization and evaluation.

### Dataset Columns

| Column                   | Description                                  | Used in Project         |
| ------------------------ | -------------------------------------------- | ----------------------- |
| `Id`                     | Unique ID of the review                      | No                      |
| `ProductId`              | Unique ID of the product                     | No                      |
| `UserId`                 | Unique ID of the reviewer                    | No                      |
| `ProfileName`            | Name of the reviewer                         | No                      |
| `HelpfulnessNumerator`   | Number of users who found the review helpful | No                      |
| `HelpfulnessDenominator` | Total number of users who rated helpfulness  | No                      |
| `Score`                  | Product rating given by the customer         | No                      |
| `Time`                   | Time of the review                           | No                      |
| `Summary`                | Short human-written summary of the review    | **Yes — Evaluation**    |
| `Text`                   | Full customer review                         | **Yes — Summarization** |

### Data Used for NLP

The two most important columns are:

#### `Text`

Contains the complete customer review.

Example:

```text
I have bought several of the vitality canned dog food products
and have found them all to be of good quality. The product looks
more like a stew than a processed meat and it smells better.
```

This is the **input text** given to the summarization system.

#### `Summary`

Contains the short summary written by a human.

Example:

```text
Good Quality Dog Food
```

This is used as a **reference summary** when evaluating the generated summaries using ROUGE.

---

## 4. Project Structure

```text
Text-Summarization-NLP/
│
├── Text-Summarization.py
│   └── Main NLP project
│
├── app.py
│   └── Streamlit interactive dashboard
│
├── Reviews.csv
│   └── Amazon Fine Food Reviews dataset
│
├── README.md
│   └── Project documentation
│
└── screenshots/
    └── Dashboard screenshots
```

---

## 5. Methods

### 5.1 Data Loading

The `Reviews.csv` dataset is loaded using Pandas.

The `Text` and `Summary` columns are used as the main NLP data.

### 5.2 Handling Missing Values

Rows with missing values in the `Summary` column are removed because the human-written summary is required for evaluation.

### 5.3 Text Preprocessing

The review text is converted to lowercase.

Sentence punctuation is preserved because it is required for sentence tokenization.

### 5.4 Sentence Tokenization

The review is divided into individual sentences using **NLTK's `sent_tokenize()`**.

For example:

```text
Original Review
      ↓
Sentence 1
Sentence 2
Sentence 3
```

### 5.5 TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to calculate the importance of words within the sentences.

The TF-IDF values are then combined to calculate an importance score for each sentence.

### 5.6 Position Score

Sentence position is also considered.

Earlier sentences receive a slightly higher position score because important information in reviews can sometimes appear near the beginning.

### 5.7 Combined Score

The final sentence score is calculated using:

```text
Final Score =
0.8 × TF-IDF Score
+
0.2 × Position Score
```

The sentence with the highest final score is selected as the generated summary.

### 5.8 Extractive Summarization

The selected sentence is taken directly from the original review.

The system does **not** generate new sentences.

### 5.9 Model Evaluation

The generated summaries are evaluated against the human-written `Summary` column using:

* ROUGE-1
* ROUGE-2
* ROUGE-L

The experiments were performed on a sample of **100 reviews**.

---

## 6. Results

Several summarization experiments were tested using the same 100-review sample.

| Experiment          |                          ROUGE-1 |         ROUGE-2 |         ROUGE-L |
| ------------------- | -------------------------------: | --------------: | --------------: |
| TF-IDF Sum + Top 2  |                           0.0996 |          0.0311 |          0.0903 |
| TF-IDF Sum + Top 1  |                           0.1168 |          0.0416 |          0.1056 |
| TF-IDF Mean + Top 2 |                           0.0996 |          0.0311 |          0.0903 |
| TF-IDF + Position   | *To be updated after experiment* | *To be updated* | *To be updated* |

### Evaluation Metrics

**ROUGE-1** measures overlap of individual words between the generated and reference summaries.

**ROUGE-2** measures overlap of two-word sequences.

**ROUGE-L** measures similarity based on the longest common subsequence.

The current baseline experiment, **TF-IDF Sum + Top 1**, produced:

```text
ROUGE-1: 0.1168
ROUGE-2: 0.0416
ROUGE-L: 0.1056
```

The TF-IDF + Position experiment will be compared with this baseline before selecting the final method.

---

## 7. Findings

The project produced the following findings:

1. **Extractive summarization can identify important sentences** from customer reviews using simple NLP techniques.

2. **TF-IDF provides a useful sentence-importance signal** by measuring the importance of words within the review.

3. Selecting **one sentence** performed better than selecting two sentences in the tested 100-review sample according to the ROUGE scores.

4. The generated summaries can be longer than human-written summaries because the system selects complete original sentences rather than generating shorter text.

5. The human-written `Summary` is often very short, while an extractive system may select a complete sentence containing more words.

6. Adding **sentence position** provides another signal that can be combined with TF-IDF. Its effectiveness should be determined from the ROUGE comparison rather than assumed.

7. The project demonstrates a complete beginner-level NLP pipeline from raw customer reviews to an interactive summarization application.

---

## 8. Tech Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Natural Language Processing

* NLTK
* Scikit-learn
* TF-IDF

### Evaluation

* ROUGE

### Visualization / Development

* Matplotlib
* Seaborn

### Dashboard

* Streamlit

### Dataset

* Amazon Fine Food Reviews Dataset

---

## 9. Getting Started

### Step 1 — Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project folder:

```bash
cd Text-Summarization-NLP
```

### Step 2 — Install Required Libraries

```bash
pip install pandas numpy nltk scikit-learn rouge-score streamlit matplotlib seaborn
```

### Step 3 — Run the Main NLP Project

```bash
python Text-Summarization.py
```

The main file performs:

```text
Data Loading
     ↓
Data Understanding
     ↓
Preprocessing
     ↓
Sentence Tokenization
     ↓
TF-IDF
     ↓
Sentence Scoring
     ↓
Summarization
     ↓
ROUGE Evaluation
```

### Step 4 — Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in the browser.

### Step 5 — Enter a Review

Paste a customer review into the text box and click:

```text
Generate Summary
```

The application will:

```text
Input Review
     ↓
Sentence Tokenization
     ↓
TF-IDF Score
     +
Position Score
     ↓
Final Score
     ↓
Most Important Sentence
     ↓
Generated Summary
```

---

## 10. Project Status

**Status:** Completed beginner-level NLP project with an interactive Streamlit dashboard.

The project currently demonstrates **extractive text summarization using TF-IDF-based sentence scoring**, with sentence position being tested as a model improvement.

Future improvements may include more advanced summarization methods such as **TextRank, BERT-based summarization, or Transformer-based abstractive summarization**.

---

## Author

**Shakeela Batool**

BS Mathematics
Namal University

---

## License

This project is created for educational and portfolio purposes.
