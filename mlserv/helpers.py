# Importing for the models

class Model:
    def __init__(self):
        import joblib, string, pickle
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC, LinearSVR
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression

        path = 'mlserv/models'
        self.vectorizer = pickle.load(open(path + '/vectorizer.pkl', 'rb'))
        self.linearSVR = joblib.load(path + '/linearSVR.joblib')
        self.linearSVC = joblib.load(path + '/lsvc.joblib')
        self.multinomialNB = joblib.load(path + '/mnb.joblib')
        self.oneVsRest = joblib.load(path + '/ovr.joblib')
        self.ensemble = joblib.load(path + '/ensemble.joblib')

        self.models = [self.linearSVC, self.multinomialNB, self.oneVsRest]
