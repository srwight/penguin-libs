'''
--- This file is for training and dumping several models as joblib files. ---
  - It's currently more of a structure. Please leave the comments in.
  - Edit the rest of the code as you see fit.
  - Put in @author: <your name> for your model section.
'''

import joblib, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd # We can remove pandas later
from sklearn.linear_model import LogisticRegression # Used in ML_stack_ensemble
from ML_stack_ensemble import *
from nltk.corpus import stopwords # Used in preprocess (for now)
from preprocessing import preprocessing
# Below imports are for the preprocessing file.
from demoticon import demotify
from cleaning_data import clean_d
from preprocess import preprocess
### Import your sklearn model ###
# Example: from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


  
# confusion_matrix(y_test, predictions) # confusion metrics 
# classification_report(y_test, predictions) # report of precision, recall, f1-score 
# accuracy_score(y_test, predictions)  
# text_classifier.predict(["Very nice movie"]) # Checking how it works - > 1 (positive sentiment) 
#####################################################################################################################################

# This path will be changed later.
path = r'C:\Users\user\Desktop\Revature\Projects\Yelp\stacked'

df = pd.read_csv(path + r'\english100k.csv')

### Vectorizer ###
''' Call the transform method before training or making predictions. '''
vectorizer = TfidfVectorizer(preprocessor = preprocessing, min_df=2)
vectorizer = vectorizer.fit(df.text)

### Place your model below with comments and parameters ###

# This model's default hyperparameters were already optimal for our data.
lsvc = LinearSVC()

#Michael Sriqui, I found 0.2 as the alpha yielded the best results 
mnb=MultinomialNB(alpha=0.2)
##################################################### GEORGE A #############################################################################
# Parameters [ instead of solver = 'sag', used solver = 'liblinear'] - based on sklearn documentations liblinear better for smaller datasets
# 'sag' for larger ones 
text_classifier = ('clf', OneVsRestClassifier(LogisticRegression(solver='sag')) # sag - stochastic average gradeint 


models = [mnb,lsvc, text_classifier] # Add each model's variable name to the list.


### Train-test split, vectorizing before making predictions ###
x = df.text
y = df.stars
x = vectorizer.transform(x)

# The below binning is just for classification models. Additional work is needed for regression.
y.replace([1, 2], -1, inplace=True)
y.replace(3, 0, inplace=True)
y.replace([4,5], 1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)


### Initial tests and dumps ###
mods = 1                   
for m in models:
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    joblib.dump(m, f"{path}\{mods}.joblib")
    mods += 1
    print(m, '\n')
    print(accuracy_score(y_test, predictions))



ensemble = fit_stack(models, x_train, y_train)
predictions = stacked_prediction(models, ensemble, x_test)
accuracy = accuracy_score(y_test, predictions)
print(prediction[0:20])
print(f"Stacked accuracy: {accuracy}")
joblib.dump(ensemble, f"{path}\ensemble.joblib")
