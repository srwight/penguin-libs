'''
--- This file is for training and dumping several models as joblib files. ---
  - It's currently more of a structure. Please leave the comments in.
  - Edit the rest of the code as you see fit.
  - Put in @author: <your name> for your model section.
'''

import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd # We can remove pandas later
from ML_stack_ensemble import *

### Import your sklearn model ###
# Example: from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVR
### Import Preprocessing here ###
# We should put all of our preprocessing together as one function.
# Example:
from preprocess import preprocess

# This path will be changed later.
path = r'C:\Users\user\Desktop\Revature\Projects\Yelp\stacked'

df = pd.read_csv(path + r'\english100k.csv')

### Vectorizer ###
''' Call the transform method before training or making predictions. '''
vectorizer = TfidfVectorizer(preprocessor = preprocess)
vectorizer = vectorizer.fit(df.text)

### Place your model below with comments and parameters ###




models = [] # Add each model's variable name to the list.


### Train-test split, vectorizing before making predictions ###
x = df.text
y = df.stars
x = vectorizer.transform(x)

# The below binning is just for classification models. Additional work is needed for regression.
y.replace([1, 2], -1)
y.replace(3, 0)
y.replace([4,5], 1)

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
