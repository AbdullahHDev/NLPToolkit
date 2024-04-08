from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Function for tokenization 
def tokenize(data, column):
    data[column] = data[column].apply(word_tokenize)
    return data

# Function for removing stopwords
def remove_stopwords(data, column):
    stop_words = set(stopwords.words('english'))
    data[column] = data[column].apply(lambda x: [word for word in x if word not in stop_words])
    return data
