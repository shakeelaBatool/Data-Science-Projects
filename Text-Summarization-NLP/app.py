import streamlit as st

from text_summarization import summarize_text


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📝 AI Text Summarizer")

st.write(
    "Enter an article or long piece of text and "
    "the Transformer model will generate a concise summary."
)


# =========================================================
# TEXT INPUT
# =========================================================

article = st.text_area(
    "Enter your article:",
    height=350,
    placeholder="Paste your article here..."
)


# =========================================================
# SUMMARY LENGTH
# =========================================================

summary_length = st.selectbox(
    "Choose summary length:",
    [
        "Short",
        "Medium",
        "Long"
    ]
)


# =========================================================
# BUTTON
# =========================================================

if st.button("Generate Summary"):

    if not article.strip():

        st.warning(
            "Please enter some text first."
        
        )
    elif len(article.split())<20:
        st.warning("Please enter a longer article atleast length of 20 words")

    else:

        # Choose length
        if summary_length == "Short":

            min_length = 20
            max_length = 70

        elif summary_length == "Medium":

            min_length = 30
            max_length = 130

        else:

            min_length = 50
            max_length = 180


        # Generate summary
        with st.spinner(
            "Generating summary..."
        ):

            summary = summarize_text(
                article,
                min_length=min_length,
                max_length=max_length
            )


        # Display result
        st.subheader("Generated Summary")

        st.success(summary)


        # Statistics
        original_words = len(
            article.split()
        )

        summary_words = len(
            summary.split()
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Original Words",
                original_words
            )


        with col2:

            st.metric(
                "Summary Words",
                summary_words
            )


        with col3:

            if original_words > 0:

                reduction = (
                    1 -
                    summary_words /
                    original_words
                ) * 100

                st.metric(
                    "Reduction",
                    f"{reduction:.1f}%"
                )