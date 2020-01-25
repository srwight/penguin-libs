from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer(TfidfVectorizer):
    def retrain(self, fp):
        data = []
        with open(fp, 'rb') as fl:
            for i, line in enumerate(fl):
                data.append(line)
        self.vocabulary_ = dict(self.vocabulary_, **self.fit(data).vocabulary_)