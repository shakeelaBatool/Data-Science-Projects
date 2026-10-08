import streamlit as st
import joblib
import re

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

model = joblib.load("toxic_comment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
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


# ============================================================
# APP TITLE
# ============================================================

st.title("💬 Toxic Comment Classifier")

st.write(
    "Enter an online comment and the NLP model will classify it "
    "as Toxic or Non-Toxic."
)


# ============================================================
# TEXT INPUT
# ============================================================

comment = st.text_area(
    "Enter your comment:",
    placeholder="Type a comment here..."
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Classify Comment"):

    if comment.strip() == "":
        st.warning("Please enter a comment first.")

    else:

        # Clean text
        cleaned_comment = clean_text(comment)

        # Convert text into TF-IDF
        comment_tfidf = vectorizer.transform(
            [cleaned_comment]
        )

        # Make prediction
        prediction = model.predict(comment_tfidf)[0]

        # Display result
        if prediction == 1:

            st.error("⚠️ Toxic Comment")

        else:

            st.success("✅ Non-Toxic Comment")


# ============================================================
# PROJECT INFORMATION
# ============================================================
