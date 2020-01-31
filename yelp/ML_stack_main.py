'''
--- This file is for training and dumping several models as joblib files. ---
'''

import joblib, string, pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from ML_stack_ensemble import *
from nltk.corpus import stopwords # Used in preprocess (for now)

# Below imports are for the preprocessing file.
from demoticon import demotify
from cleaning_data import clean_d
from preprocess import preprocess
from preprocessing import preprocessing

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD


 
#####################################################################################################################################

# This path will be changed later.
path = r'C:\Users\user\Desktop\Revature\Projects\Yelp\stacked'

df = pd.read_csv(path + r'\english100k.csv')

### Vectorizer ###
''' We call the transform method before training or making predictions. '''
vectorizer = TfidfVectorizer(preprocessor = preprocessing, ngram_range = (1,2), min_df=3)
vectorizer = vectorizer.fit(df.text)

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

### Models ###

# This model's default hyperparameters were already optimal for our data.
lsvc = CalibratedClassifier(LinearSVC())

# Two parameters need to be changed, but only because they work better when records > features, and that will be our scenario.
linearSVR = LinearSVR(loss='squared_epsilon_insensitive', dual=False)

#Michael Sriqui, I found 0.2 as the alpha yielded the best results 
mnb=MultinomialNB(alpha=0.2)

##################################################### GEORGE A #############################################################################
# Parameters [ instead of solver = 'sag', used solver = 'liblinear'] - based on sklearn documentations liblinear better for smaller datasets
# 'sag' for larger ones 
text_classifier = OneVsRestClassifier(LogisticRegression(solver='sag')) # sag - stochastic average gradeint 


models = [mnb, lsvc, text_classifier] # Classifier models for the ensemble.

LSA = TruncatedSVD(n_components = 100)

### Train-test split, vectorizing before making predictions ###
x = df.text
y = df.stars
x = vectorizer.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)

### Latent Semantic Analysis ###
LSA.fit_transform(x_train)
joblib.dump(LSA, f"lsa.joblib")
LSA.transform(x_test)

linearSVR.fit(x_train, y_train) #We fit the SVR before binning for classifiers.
joblib.dump(linearSVR, "linearSVR.joblib")

# The below binning is for classification models (negative, neutral, positive).
# Moved replace to after linearSVR is fitted.
y_train.replace([1, 2], -1, inplace=True)
y_train.replace(3, 0, inplace=True)
y_train.replace([4,5], 1, inplace=True)
y_test.replace([1, 2], -1, inplace=True)
y_test.replace(3, 0, inplace=True)
y_test.replace([4,5], 1, inplace=True)



### Fitting, testing, and dumping the classifiers ###
mods = 1                   
for m in models:
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    joblib.dump(m, f"{path}\{mods}.joblib")
    mods += 1
    tests = []
    for x in predictions:
      y = np.argmax(x)
      y -= 1
      tests.append(y)
    print(m, '\n')
    precision = precision_score(y_test, tests, average='micro')
    recall = recall_score(y_test, tests, average='micro')
    f1_score = 2 * (precision * recall) / (precision + recall)
    print( "Precision: " + str(precision) + " Recall: " + str(recall) + " F1 Score: " + str(f1_score))
    print(confusion_matrix(y_test, tests)
      
    print(accuracy_score(y_test, tests))


### Fitting, testing, and dumping the stacking ensemble ###
ensemble = fit_stack(models, x_train, y_train)
predictions = stacked_prediction(models, ensemble, x_test)
accuracy = accuracy_score(y_test, predictions)
print(predictions[0:20])
print(f"Stacked accuracy: {accuracy}")
joblib.dump(ensemble, f"{path}\ensemble.joblib")

precision = precision_score(y_test, predictions, average='micro')
recall = recall_score(y_test, predictions, average='micro')
f1_score = 2 * (precision * recall) / (precision + recall)
print( "Precision: " + str(precision) + " Recall: " + str(recall) + " F1 Score: " + str(f1_score))
