import os
import pandas as pd
from rouge_score import rouge_scorer

from text_summarization import summarize_text


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = "data/articles.csv"

RESULTS_DIR = "results"

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "evaluation_results.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

if "article" not in df.columns:

    raise ValueError(
        "CSV must contain an 'article' column."
    )


if "summary" not in df.columns:

    raise ValueError(
        "CSV must contain a 'summary' column."
    )


# =========================================================
# ROUGE SCORER
# =========================================================

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)


# =========================================================
# EVALUATION
# =========================================================

results = []

total_rouge1 = 0
total_rouge2 = 0
total_rougeL = 0


print("\nStarting evaluation...")


for index, row in df.iterrows():

    article = str(row["article"])

    reference = str(row["summary"])


    # Generate summary
    prediction = summarize_text(article)


    # Calculate ROUGE
    scores = scorer.score(
        reference,
        prediction
    )


    rouge1 = scores["rouge1"].fmeasure
    rouge2 = scores["rouge2"].fmeasure
    rougeL = scores["rougeL"].fmeasure


    total_rouge1 += rouge1
    total_rouge2 += rouge2
    total_rougeL += rougeL


    results.append({

        "article": article,

        "reference_summary": reference,

        "generated_summary": prediction,

        "rouge1": rouge1,

        "rouge2": rouge2,

        "rougeL": rougeL
    })


    print(
        f"Article {index + 1}/{len(df)} completed"
    )


# =========================================================
# AVERAGE SCORES
# =========================================================

number_of_articles = len(df)


average_rouge1 = total_rouge1 / number_of_articles
average_rouge2 = total_rouge2 / number_of_articles
average_rougeL = total_rougeL / number_of_articles


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n")
print("=" * 50)
print("FINAL ROUGE RESULTS")
print("=" * 50)

print(
    f"ROUGE-1: {average_rouge1:.4f}"
)

print(
    f"ROUGE-2: {average_rouge2:.4f}"
)

print(
    f"ROUGE-L: {average_rougeL:.4f}"
)


# =========================================================
# SAVE RESULTS
# =========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


results_df = pd.DataFrame(results)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


print("\nResults saved to:")
print(RESULTS_PATH)