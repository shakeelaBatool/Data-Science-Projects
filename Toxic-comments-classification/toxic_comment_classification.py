# ============================================================
# TOXIC COMMENT CLASSIFICATION - NLP PROJECT
# ============================================================

import pandas as pd
import re
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

data = pd.read_csv("data_set/comments.csv")

print("Dataset loaded successfully!")

# ------------------------------------------------------------
# 1. SHOW FIRST 5 ROWS
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(data.head())

# ------------------------------------------------------------
# 2. DATASET SHAPE
# ------------------------------------------------------------

print("\nDataset Shape:")
print(data.shape)

# ------------------------------------------------------------
# 3. COLUMN NAMES
# ------------------------------------------------------------

print("\nColumn Names:")
print(data.columns)

# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values:")
print(data.isnull().sum())

# ------------------------------------------------------------
# 5. DUPLICATE ROWS
# ------------------------------------------------------------

print("\nDuplicate Rows:")
print(data.duplicated().sum())

# ------------------------------------------------------------
# 6. LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\nLabel Distribution:")
print(data["label"].value_counts())


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 1. VISUALIZE LABEL DISTRIBUTION - PIE CHART
# ------------------------------------------------------------

label_counts = data["label"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    label_counts,
    labels=["Non-Toxic (0)", "Toxic (1)"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Toxic vs Non-Toxic Comments")
plt.show()


# ============================================================
# STEP 3: TEXT PREPROCESSING
# ============================================================

# ------------------------------------------------------------
# 1. TEXT CLEANING FUNCTION
# ------------------------------------------------------------

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags such as <br>
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove newline characters
    text = re.sub(r"\n", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


# ------------------------------------------------------------
# 2. APPLY TEXT CLEANING
# ------------------------------------------------------------

data["clean_text"] = data["text"].apply(clean_text)


# ------------------------------------------------------------
# 3. DISPLAY ORIGINAL VS CLEANED TEXT
# ------------------------------------------------------------

print("\nOriginal vs Cleaned Text:\n")

for i in range(6):

    print("ORIGINAL:")
    print(data["text"].iloc[i])

    print("\nCLEANED:")
    print(data["clean_text"].iloc[i])

    print("\n" + "=" * 80 + "\n")


# ============================================================
# STEP 4: TOKENIZATION
# ============================================================

import nltk

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize


# ------------------------------------------------------------
# 1. TOKENIZE CLEANED TEXT
# ------------------------------------------------------------

data["tokens"] = data["clean_text"].apply(word_tokenize)


# ------------------------------------------------------------
# 2. DISPLAY TOKENIZATION EXAMPLES
# ------------------------------------------------------------

print("\nTokenization Examples:\n")

for i in range(3):

    print("Cleaned Text:")
    print(data["clean_text"].iloc[i])

    print("\nTokens:")
    print(data["tokens"].iloc[i])

    print("\n" + "=" * 80)


# ============================================================
# STEP 5: STOPWORD ANALYSIS
# ============================================================

from nltk.corpus import stopwords

# Download English stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


# ------------------------------------------------------------
# 1. REMOVE STOPWORDS FOR ANALYSIS
# ------------------------------------------------------------

data["tokens_without_stopwords"] = data["tokens"].apply(
    lambda tokens: [
        word for word in tokens
        if word not in stop_words
    ]
)


# ------------------------------------------------------------
# 2. DISPLAY STOPWORD REMOVAL EXAMPLES
# ------------------------------------------------------------

print("\nStopword Removal Examples:\n")

for i in range(3):

    print("Original Tokens:")
    print(data["tokens"].iloc[i])

    print("\nAfter Stopword Removal:")
    print(data["tokens_without_stopwords"].iloc[i])

    print("\n" + "=" * 80)


# ============================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. SELECT INPUT AND TARGET
# ------------------------------------------------------------

X = data["clean_text"]
y = data["label"]


# ------------------------------------------------------------
# 2. SPLIT THE DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 3. CHECK THE SPLIT
# ------------------------------------------------------------

print("\nTrain-Test Split:")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())


# ============================================================
# STEP 7: TF-IDF VECTORIZATION
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------
# 1. CREATE TF-IDF VECTORIZER
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000
)


# ------------------------------------------------------------
# 2. FIT TF-IDF ON TRAINING DATA
# ------------------------------------------------------------

X_train_tfidf = vectorizer.fit_transform(X_train)


# ------------------------------------------------------------
# 3. TRANSFORM TEST DATA
# ------------------------------------------------------------

X_test_tfidf = vectorizer.transform(X_test)


# ------------------------------------------------------------
# 4. CHECK THE RESULT
# ------------------------------------------------------------

print("\nTF-IDF Vectorization:")

print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

print("\nNumber of TF-IDF features:")
print(len(vectorizer.get_feature_names_out()))

print("\nFirst 20 TF-IDF features:")
print(vectorizer.get_feature_names_out()[:20])
# ============================================================
# STEP 8: CLASSIFICATION - MULTINOMIAL NAIVE BAYES
# ============================================================

from sklearn.naive_bayes import MultinomialNB

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train_tfidf, y_train)

print("\nModel trained successfully!")

# Make predictions
y_pred = model.predict(X_test_tfidf)

print("Predictions completed!")
# ============================================================
# STEP 9: MODEL EVALUATION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Non-Toxic", "Toxic"]
    )
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Toxic", "Toxic"]
)

disp.plot()
plt.title("Confusion Matrix - Toxic Comment Classification")
plt.show()
# ============================================================
# STEP 10: TEST NEW COMMENTS
# ============================================================

def predict_comment(comment):

    # Clean the comment
    cleaned_comment = clean_text(comment)

    # Convert text into TF-IDF
    comment_tfidf = vectorizer.transform([cleaned_comment])

    # Predict
    prediction = model.predict(comment_tfidf)[0]

    if prediction == 1:
        return "Toxic"
    else:
        return "Non-Toxic"


# Example
comment = "Thank you for sharing this information."

result = predict_comment(comment)

print("\nNew Comment:")
print(comment)

print("Prediction:")
print(result)
# ============================================================
# STEP 11: SAVE TRAINED MODEL
# ============================================================

import joblib

joblib.dump(model, "toxic_comment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("toxic_comment_model.pkl")
print("tfidf_vectorizer.pkl")