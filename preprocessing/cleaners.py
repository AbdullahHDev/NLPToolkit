import pandas as pd
import re

# Function for cleaning text data
def text_cleaning(data, column, lowercase=True, remove_urls=True, remove_non_word=True, remove_digits=True):
    if lowercase:
        data[column] = data[column].str.lower()
    if remove_urls:
        data[column] = data[column].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", '', x, flags=re.MULTILINE))
    if remove_non_word:
        data[column] = data[column].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    if remove_digits:
        data[column] = data[column].apply(lambda x: re.sub(r'\d+', '', x))
    return data
