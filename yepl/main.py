import json, pandas as pd
data = []
with open('yelp_academic_dataset_review.json', 'r', encoding="utf8") as fl:
    for line in range(1, 500000):
        print(f'Reading line {line}')
        data.append(fl.readline())
    data = list(map(json.loads, data))
    df = pd.DataFrame(data)
df.to_csv('yelp_reviews.csv')

# Cleaning Data
from cleaning data import clean_d
text_modified = []
stars_modified = []
for x in data:
    for a,y in x.items():
        if a == 'stars':
            stars_modified.append(y)
        if a == 'text':
            text_modified.append(clean_d(y))
list_of_tuples = list(zip(stars_modified, text_modified)) 


df2 = pd.DataFrame(list_of_tuples, columns = ['stars', 'text']) 

# COUNTING MISSING VALUES IF THEY EXIT 
df2.isnull().sum()
# GOOD TO GO! 

# Count # of each labels
df2['stars'].value_counts()

# TRAIN DATA FEATURES 
X = df2['text']

# TEST DATA 
y = df2['stars']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))




# ONLY training set got vecorized (turned into vocabulary) so we need to turn 
# test set to vocabulary as well. However, better alternative is to use Pipeline Class , which 
# can perform both vectorization and classification with one shot 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
text_classifier  = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])


text_classifier.fit(X_train, y_train)


predictions = text_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


print(confusion_matrix(y_test, predictions))


print(accuracy_score(y_test, predictions))


# Checking some texts 
text_classifier.predict(["bad but its good!"])