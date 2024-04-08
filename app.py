import streamlit as st
from preprocessing import cleaners, tokenization, lemmatization
from utils import download_utils
import pandas as pd
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_options(data):
    st.sidebar.header("Preprocessing Options")
    text_column = st.sidebar.selectbox("Select text column", data.columns)
    
    # Text cleaning options
    st.sidebar.subheader("Text Cleaning")
    lowercase = st.sidebar.checkbox("Convert to lowercase")
    remove_urls = st.sidebar.checkbox("Remove URLs")
    remove_non_word = st.sidebar.checkbox("Remove non-word characters")
    remove_digits = st.sidebar.checkbox("Remove numerical digits")

    # Tokenization and stopwords options
    st.sidebar.subheader("Tokenization")
    tokenize_text = st.sidebar.checkbox("Tokenize text")
    remove_stopwords = st.sidebar.checkbox("Remove stopwords")

    # Stemming and lemmatization options
    st.sidebar.subheader("Stemming and Lemmatization")
    apply_stemming = st.sidebar.checkbox("Apply stemming")
    apply_lemmatization = st.sidebar.checkbox("Apply lemmatization")

    if st.sidebar.button("Apply Preprocessing"):
        if lowercase or remove_urls or remove_non_word or remove_digits:
            data = cleaners.text_cleaning(data, text_column, lowercase, remove_urls, remove_non_word, remove_digits)
        if tokenize_text:
            data = tokenization.tokenize(data, text_column)
        if remove_stopwords:
            data = tokenization.remove_stopwords(data, text_column)
        if apply_stemming:
            data = lemmatization.stemming(data, text_column)
        if apply_lemmatization:
            data = lemmatization.lemmatization(data, text_column)
        
        st.write("Preprocessed Data")
        st.dataframe(data)
        
        tmp_download_link = download_utils.download_link(data, 'preprocessed_text_data.csv', 'Download preprocessed text data')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

def main():
    st.title("CSV Text Preprocessing App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Original Data")
        st.dataframe(data)
        
        preprocess_options(data)

if __name__ == "__main__":
    main()
