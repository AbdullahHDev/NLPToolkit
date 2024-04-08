from nltk.stem import PorterStemmer, WordNetLemmatizer

# Function for stemming
def stemming(data, column):
    stemmer = PorterStemmer()
    data[column] = data[column].apply(lambda x: [stemmer.stem(word) for word in x])
    return data

# Function for lemmatization
def lemmatization(data, column):
    lemmatizer = WordNetLemmatizer()
    data[column] = data[column].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return data
