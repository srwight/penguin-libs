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
##################################################### GEORGE A #############################################################################
# Parameters [ instead of solver = 'sag', used solver = 'liblinear'] - based on sklearn documentations liblinear better for smaller datasets
# 'sag' for larger ones 
# 
# Importing all dependencies and libraries here some of them are imported above 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline # Feeding into the pipelne so it does Counte Vectorizer, TF Transformer TFIFD 
from sklearn.feature_extraction.text import TfidfVectorizer
from cleaning_data import clean_d # Custom Made Reg Expressions Text Cleaning Took 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

# Checling ,issing values 
df.isnull().sum()

# WE GET 5 unique values
df['stars'].unique()

# counts how many of each individual label is in the dataset (column stars)
df['stars'].value_counts()

 # Replacing 2 with 1 and 4 with 5 (labels number of stars)
df['stars'] = df['stars'].replace({2: 1, 4: 5})

# Replacing 1,3,5 with -1, 0 and 1 SO would reflect Softmax probabilities representations function 
df['stars'] = df['stars'].replace({1:-1, 3:0, 5:1})

# counts number of unique labels in the stars columns 
df['stars'].unique()

# Assigning 'text' column to X
X = df['text']

# Assigning 'stars' values to be y 
y = df['stars']

# Splitting Data into Training and testing  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
 
stop_words = set(stopwords.words('english'))
text_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag')), # sag - stochastic average gradeint 
            ])
# text_classifier.fit(X_train, y_train) - training data (its saved in george_model_joblib
# Prediction using predict method 
predictions = text_classifier.predict(X_test)
  
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
vectorizer = TfidfVectorizer(preprocessor = preprocessing)
vectorizer = vectorizer.fit(df.text)

### Place your model below with comments and parameters ###
# This model's default hyperparameters were already optimal for our data.
lsvc = LinearSVC()



models = [lsvc] # Add each model's variable name to the list.


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
