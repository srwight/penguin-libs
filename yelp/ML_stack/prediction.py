import joblib, string, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from ML_stack_ensemble import *
from preprocessing import preprocessing
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

vectorizer = pickle.load('vectorizer.pkl')
linearSVR = joblib.load('linearSVR.joblib')
linearSVC = joblib.load('1.joblib')
multinomialNB = joblib.load('2.joblib')
oneVsRest = joblib.load('3.joblib')
ensemble = joblib.load('ensemble.joblib')

models = [linearSVC, multinomialNB, oneVsRest]

review = input()
review = vectorizer.transform(review)

prediction = stacked_prediction(models, ensemble, review)

linear_predict = linearSVR.predict(review)

if prediction == 1:
    print('Positive: ')
elif prediction == 0:
    print('Neutral: ')
else:
    print('Negative: ')

print(linear_predict)