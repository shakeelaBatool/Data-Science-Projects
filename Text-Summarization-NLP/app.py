import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK sentence tokenizer
nltk.download("punkt_tab", quiet=True)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Text Summarization NLP",
    page_icon="📝",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .summary-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f7f7f7;
        font-size: 18px;
        line-height: 1.6;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f2f2f2;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">TEXT SUMMARIZATION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP-Based Extractive Summarization Dashboard</div>',
    unsafe_allow_html=True
)

st.write(
    "This dashboard uses Natural Language Processing (NLP) "
    "and TF-IDF to extract the most important sentence from "
    "a customer review."
)


# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Customer Review")

text = st.text_area(
    "Paste your review below:",
    height=220,
    placeholder="Example: I bought this product recently and I really like it..."
)


# -----------------------------
# Summarization
# -----------------------------
if st.button("Generate Summary", type="primary"):

    # Check empty input
    if text.strip() == "":
        st.warning("Please enter a customer review first.")

    else:

        # Convert text to lowercase
        text = text.lower()

        # Split text into sentences
        sentences = sent_tokenize(text)

        # Check number of sentences
        if len(sentences) == 0:
            st.warning("No valid sentences were found.")

        elif len(sentences) == 1:

            summary = sentences[0]

            st.subheader("Generated Summary")

            st.markdown(
                f'<div class="summary-box">{summary}</div>',
                unsafe_allow_html=True
            )

            st.info("The review contains only one sentence, so it was returned as the summary.")

        else:

            # -----------------------------
            # TF-IDF
            # -----------------------------
            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            tfidf_matrix = vectorizer.fit_transform(sentences)

            # -----------------------------
            # Sentence Scores
            # TF-IDF Sum
            # -----------------------------
            sentence_scores = tfidf_matrix.sum(axis=1).A1

            # -----------------------------
            # Select Top 1 Sentence
            # -----------------------------
            top_index = sentence_scores.argmax()

            summary = sentences[top_index]


            # -----------------------------
            # Output
            # -----------------------------
            st.subheader("Generated Summary")

            st.markdown(
                f'<div class="summary-box">{summary}</div>',
                unsafe_allow_html=True
            )


            # -----------------------------
            # Information
            # -----------------------------
            st.subheader("Summary Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Original Sentences",
                    len(sentences)
                )

            with col2:
                st.metric(
                    "Summary Sentences",
                    1
                )

            with col3:
                st.metric(
                    "Method",
                    "TF-IDF"
                )


            # -----------------------------
            # Sentence Scores
            # -----------------------------
            st.subheader("Sentence Importance Scores")

            for i, score in enumerate(sentence_scores):

                if i == top_index:

                    st.success(
                        f"Sentence {i + 1} — Score: {score:.4f} ⭐\n\n"
                        f"{sentences[i]}"
                    )

                else:

                    st.write(
                        f"Sentence {i + 1} — Score: {score:.4f}"
                    )

                    st.write(sentences[i])


# -----------------------------
# About Section
# -----------------------------
st.divider()

st.subheader("About This Project")

st.write(
    "This project demonstrates extractive text summarization using NLP. "
    "The system splits a review into sentences, calculates TF-IDF scores "
    "for the sentences, and selects the sentence with the highest total "
    "TF-IDF score as the generated summary."
)

st.write(
    "**Final Model:** TF-IDF Sum + Top 1 Sentence"
)

st.write(
    "**Evaluation:** ROUGE-1, ROUGE-2, and ROUGE-L"
)
