from sklearn.feature_extraction.text import TfidfVectorizer
import json, warnings
warnings.simplefilter("ignore", UserWarning)

class Vectorizer(TfidfVectorizer):
    def train(self, fp, samples:int=None, seeker:int=None, debug:bool=False):
        """
        Replacement for the `fit` method of the TfidfVectorizer parent class.
        This method accepts a json file as input and has optional `samples` and `seeker` parameters.

        Parameters
        ----------
        fp : str
            File path of the desired file to train on.

        samples : int (default=None)
            Amount of lines of the given file to read through.
            
        seeker : int (default=None)
            Line number to start on for reading lines.

        debug : bool (default=False)
            If set to true, prints status of file enumerator every 10,000 lines and when fitting data.

        Examples
        --------
        >>> from vectorizer import Vectorizer
        >>> vec = Vectorizer()
        >>> vec.train('data1.json')
        >>>
        >>> print(len(vec.vocabulary_))
        28391
        >>> vec.train('data2.json', samples=1000)
        >>> print(len(vec.vocabulary_))
        31923
        >>> vec.train('data2.json', samples=4000, seeker=1000)
        >>> print(len(vec.vocabulary_))
        37189
        """
        if fp.endswith('.json'):  
            data = []
            with open(fp, 'r') as fl:
                if seeker:
                    for i, line in enumerate(fl):
                        if i < seeker - 1:
                            continue
                        else:
                            break
                for i, line in enumerate(fl):
                    if debug:
                        if i % 10000 == 0:
                            print(f'\rAppending line {i}', end='')
                    if samples:
                        if i < samples:
                            data.append(json.loads(line)['text'])
                        else:
                            break
                    else:
                        data.append(json.loads(line)['text'])
            if debug:
                print('\nFitting data...')
            if hasattr(self, 'vocabulary_'):
                self.vocabulary_ = dict(self.vocabulary_, **self.fit(data).vocabulary_)
            else:
                self.fit(data)
        else:
            print('Please provide a json file with text and stars as keys.')
