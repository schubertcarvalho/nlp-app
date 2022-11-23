"""
Created on Tue Nov 22 2022

@author: Schubert Carvalho

Required Packages: streamlit textblob spacy gensim neattext matplotlib wordcloud 
Spacy Model: python -m spacy download en_core_web_sm
"""

# Core Pkgs
import streamlit as st

st.set_page_config(
    page_title="NLP Examples",
    page_icon="tb-logo-lg.png",
    layout="centered",
    initial_sidebar_state="auto",
)

# NLP Pkgs
from textblob import TextBlob
import spacy
import neattext as nt
from langdetect import detect

# It was removed from 4.0
# from gensim.summarization import summarize

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

# Text summarization using T5
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# matplotlib.use("Agg")
# Python conflict during installation
# from wordcloud import WordCloud

import utils


@st.cache
def text_summarize(input_text):
    """Performs text summarization using T5 model"""
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)

    inputs = tokenizer.encode(
        "summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True
    )

    summary_ids = model.generate(
        inputs, max_length=256, min_length=0, length_penalty=5.0, num_beams=2
    )

    summary = tokenizer.decode(summary_ids[0])

    return summary


# Function For Tokens and Lemma Analysis
@st.cache
def text_analyzer(input_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(input_text)
    all_data = [
        ('"Token":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx
    ]
    return all_data


def main():
    """NLP App with Streamlit and TextBlob"""

    # Add title
    utils.write_title()
    utils.add_sidebar_logo()

    # Add Selectbox Menu for text analysis
    activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("NLP Tools", activity)

    # Text Analysis CHOICE
    if choice == "Text Analysis":

        st.subheader("Text Analysis")
        st.write("")
        st.write("")

        raw_text = st.text_area("Text input", "Enter a Text in English...", height=250)

        if st.button("Analyze"):
            if len(raw_text) == 0:
                st.write("Enter text ...")
            else:
                blob = TextBlob(raw_text)
                # if blob.detect_language() == "en":
                #     st.write("engl")
                st.info("Basic Features")
                basic_info_col, processed_text_col = st.columns(2)

                with basic_info_col:
                    with st.expander("Basic Info"):
                        st.success("Text Stats")
                        word_description = nt.TextFrame(raw_text).word_stats()
                        result_word_descripton = {
                            "Text Length": word_description["Length of Text"],
                            "Num of Vowels": word_description["Num of Vowels"],
                            "Num of Consonants": word_description["Num of Consonants"],
                            "Num of Stopwords": word_description["Num of Stopwords"],
                        }
                        st.write(result_word_descripton)
                    with st.expander("Stopwords"):
                        st.success("Stop Words List")
                        stop_w = nt.TextExtractor(raw_text).extract_stopwords()
                        st.error(set(stop_w))

                with processed_text_col:
                    with st.expander("Processed Text"):
                        st.success("Stopwords Excluded Text")
                        processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
                        st.write(processed_text)

                # Add spaces
                st.write("")
                st.write("")

                st.info("Advanced Features")
                token_lema_col, summarize_col = st.columns(2)
                with token_lema_col:
                    with st.expander("Tokens&Lemas"):
                        st.success("T&L")
                        processed_text_mid = str(
                            nt.TextFrame(raw_text)
                            .remove_stopwords()
                            .remove_puncts()
                            .remove_special_characters()
                        )
                        tandl = text_analyzer(processed_text_mid)
                        st.json(tandl)

                with summarize_col:
                    with st.expander("Summarize"):
                        st.success("Summarize")
                        st.write(text_summarize(raw_text))

    # Translation CHOICE
    elif choice == "Translation":
        st.subheader("Text Translation")
        # Add spaces
        st.write("")
        st.write("")

        raw_text = st.text_area("Input text", "Write text to be translated...")
        if len(raw_text) < 3:
            st.warning("Please provide a string with at least 3 characters...")
        else:
            blob = TextBlob(raw_text)
            detect_language = detect(raw_text)

            translation_options = st.selectbox(
                "Select translation language",
                ["Chinese", "English", "German", "Italian", "Russian", "Spanish"],
            )

            if st.button("Translate"):
                if translation_options == "Italian":
                    st.text("Translating to Italian...")
                    translation_result = blob.translate(
                        from_lang=detect_language, to="it"
                    )
                elif translation_options == "Spanish" and detect_language != "es":
                    st.text("Translating to Spanish...")
                    translation_result = blob.translate(
                        from_lang=detect_language, to="es"
                    )
                elif translation_options == "Chinese" and detect_language != "zh-CN":
                    st.text("Translating to Chinese...")
                    translation_result = blob.translate(
                        from_lang=detect_language, to="zh-CN"
                    )
                elif translation_options == "Russian" and detect_language != "ru":
                    st.text("Translating to Russian...")
                    translation_result = blob.translate(
                        from_lang=detect_language, to="ru"
                    )
                elif translation_options == "German" and detect_language != "de":
                    st.text("Translating to German...")
                    translation_result = blob.translate(
                        from_lang=detect_language, to="de"
                    )
                elif translation_options == "English" and detect_language != "en":
                    st.text("Translating to English...")
                    translation_result = blob.translate(
                        from_lang=detect_language, to="en"
                    )
                else:
                    translation_result = (
                        "Text is already in " + "'" + detect_language + "'"
                    )

                st.success(translation_result)

        # Sentiment Analysis CHOICE
    elif choice == "Sentiment Analysis":

        st.subheader("Sentiment Analysis")
        # Add spaces
        st.write("")
        st.write("")

        raw_text = st.text_area("Input text", "Write text here...")
        if st.button("Evaluate"):
            if len(raw_text) == 0:
                st.warning("Enter a text...")
            else:
                blob = TextBlob(raw_text)
                detected_language = detect(raw_text)

                if detected_language != "en":
                    translated_language = blob.translate(
                        from_lang=detected_language, to="en"
                    )
                    blob = TextBlob(str(translated_language))

                sentiment = blob.sentiment
                st.success("Sentiment Polarity: {}".format(sentiment.polarity))
                st.success("Sentiment Subjectivity: {}".format(sentiment.subjectivity))

    # About CHOICE
    else:  # choice == 'About':
        st.subheader("Visit TubeBuddy")

        st.write("")
        st.write("")

        st.markdown(
            """        
        
         **[TubeBuddy](https://www.tubebuddy.com/)**
        
        """
        )


if __name__ == "__main__":

    main()
