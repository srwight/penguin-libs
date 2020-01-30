''' Add demotify, regex, etc. '''

def preprocess(text):
    import string
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english')) #'if', 'and', 'the', etc.
    translation = str.maketrans('', '', string.punctuation)
    text = text.translate(translation)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text