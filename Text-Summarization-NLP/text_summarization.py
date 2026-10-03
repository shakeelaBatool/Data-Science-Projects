import re
import torch
from transformers import pipeline


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_NAME = "facebook/bart-large-cnn"

# Use GPU if available, otherwise CPU
DEVICE = 0 if torch.cuda.is_available() else -1


# =========================================================
# LOAD TRANSFORMER MODEL
# =========================================================

print("Loading Transformer model...")

summarizer = pipeline(
    "summarization",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE
)

print("Model loaded successfully!")


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(text, max_words=450):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


# =========================================================
# GENERATE SUMMARY
# =========================================================

def summarize_text(
    text,
    min_length=30,
    max_length=130
):

    text = clean_text(text)

    if not text:
        return "Please enter some text."

    # Split long articles into manageable pieces
    chunks = split_text(text)

    summaries = []

    for chunk in chunks:

        try:

            result = summarizer(
                chunk,
                min_length=min_length,
                max_length=max_length,
                do_sample=False
            )

            summaries.append(result[0]["summary_text"])

        except Exception as e:

            print("Error while summarizing:", e)

    # Combine summaries
    final_summary = " ".join(summaries)

    return final_summary