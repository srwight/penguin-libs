####################################################### ORIGINAL FIRST GIT PUSH #########################################################
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

data = []
with open('english_only_reviews.json', 'r', encoding="utf8") as fl:
    for line in range(1, 100000):
        print(f'Reading line {line}')
        data.append(fl.readline())
    data = list(map(json.loads, data))
    df = pd.DataFrame(data)
df.to_csv('yelp_reviews.csv')

 # HERE, AFTER CLEANING DATA WITH clean_d function 
text_modified = []
stars_modified = []
for x in data:
    for a,y in x.items():
        if a == 'stars':
            stars_modified.append(y)
        if a == 'text':
            text_modified.append(clean_d(y))
            
#we are converting back to lists, so we can create a dataframe out of those lists  
list_of_stars_text = list(zip(stars_modified, text_modified)) 

# Creating a dataframe with 2 columns [stars and text] 
df = pd.DataFrame(list_of_stars_text, columns = ['stars', 'text'])

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
 
# # COUNT VECTORIZER, TFIDF TRANSFORMER and FITTING THE MODEL SEPARATELY! in several steps #
# from sklearn.feature_extraction.text import CountVectorizer
# count_vector = CountVectorizer() # creating an object from the Class CountVectorizer 

# FIT VECTORIZER TO THE DATA (build a vocab, count the number of words...)
# count_vector.fit(X_train)

# # Transform the original text message ---> VECTOR
# X_train_counts = count_vector.transform(X_train)

# fit and transform in one step - 1 line (more efficient)
# X_train_counts = count_vector.fit_transform(X_train)

# NOW WE WANT TO TRANSFORM COUNTS TO FREQUENCIES WITH TFIDF 
# SO BASICALLY WE ARE FIGURING OUT WHICH WORDS ARE MORE IMPORTANT THAN OTHERS 
# GIVING WORDS THAT ARE MORE IMPORTANT MORE WEIGHTS 
# SO WHAT WE ARE GOING TO DO IS TO PASS CountVectorization to TFIDF Transformer
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# SO there is better way to do count vecorization and tfidf with 1 shot ! (TfidfVectorizer class)
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()
# X_train_Tfidf = vectorizer.fit_transform(X_train)

# NOW WE CAN TRAIN CLASSIFIER 
# from sklearn.svm import LinearSVC
# classifier = LinearSVC() # I was experimenting with different models 

# classifier.fit(X_train_Tfidf, y_train) - trained model 

# ONLY training set got vecorized (turned into vocabulary) so we need to turn 
# test set to vocabulary as well. However, better alternative is to use Pipeline Class , which 
# can perform both vectorization and classification with one shot !!! 

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
