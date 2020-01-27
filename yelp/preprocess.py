# File created by Stephen Wight

from json import loads, dumps
from math import log

## my own frequency token generator.
class eda_crawler():
    def __init__(self, stopwords:set = set()):
        self.vocab = dict()
        self.vocab_ratings = dict()
        # self.langs = dict()
        self.docs = 0
        self.stopwords = stopwords
    
    def fit(self, instring, rating):
        self.docs += 1

        # this part increments a word-in-doc count for each unique word in the document
        # as well as adding an entry for the rating of the document for each word.
        for word in set(instring.split()):
            if word in self.stopwords:
                continue
            if word in self.vocab.keys():
                self.vocab[word] += 1
                self.vocab_ratings[word].append(rating)
            else:
                self.vocab[word] = 1
                self.vocab_ratings[word] = [rating,]

    def get_tfidf_tokens(self, instring):
        doc_vocab = dict()
        for word in instring.split():
            if word in self.stopwords:
                continue
            if word in doc_vocab.keys():
                doc_vocab[word] += 1
            else:
                doc_vocab[word] = 1
        doctfidf = dict()    
        for word in doc_vocab.keys():
            if word not in self.vocab:
                continue
            doctfidf[word] = (doc_vocab[word]/len(doc_vocab)) * log(self.docs/self.vocab[word])
        return doctfidf

    def get_term_docfreq(self, term):
        if term in self.vocab.keys():
            return self.vocab[term]
        else:
            return 0

    def output_attr(self, attribute:str, sepr:str = '_', filename:str='crawler') -> str:
        filetag = f'{sepr}{attribute}'
        if filetag not in filename:
            if '.' in filename:
                if filename.rsplit('.',1)[1].lower() != 'json':
                    filename = f'{filename}{filetag}.json'
                else:
                    filename = f'{filename.rsplit(".",1)[0]}{filetag}.json'
            else:
                filename = f'{filename}{filetag}.json'    
        with open(filename, 'wt') as fl_out:
            if attribute == 'vocab':
                fl_out.write(dumps(self.vocab))
            if attribute == 'ratings':
                for i, word in enumerate(self.vocab_ratings):
                    output = dumps(f'{{ {word}:{self.vocab_ratings["word"]} }}')
                    fl_out.write(f'{output}\n')
                    print('')
                    if i % 100 == 0: print(f'\rOutputting line {i}.', end='')
            if attribute == 'docs':
                fl_out.write(dumps({'DocCount':self.docs}))
        return filename

    def get_term_ratings_list(self, term):
        if term in self.vocab_ratings.keys():
            return self.vocab_ratings[term]
        else:
            return [0,]
    

def demotify(indoc:str) -> str:
    """demotify - remove select emoticons from strings, then return the strings

    This function removes any emoticons (the precursor to graphical emoji) from 
    a string of text, replacing them with a word that represents their emotional
    content.

    Args:
        indoc: This is the string that may contain emoticons to be replaced.

    Returns:
        str: This function returns the string with emoticons replaced by words


    """

    emoticon_dict = {
        ':)': ' emhappy ',
        ':-)':' emhappy ',
        ':(': ' emsad ',
        ':-(':' emsad ',
        '>:(':' emangry ',
        ':D': ' emgrin ',
        ':-D':' emgrin ',
        ';)': ' emwink ',
        ';-)':' emwink ',
        '&'  :' and ',
    }

    for key in emoticon_dict:
        indoc = indoc.replace(key, emoticon_dict[key])

    return indoc

def getfields(line:str, *fields) -> str:
    '''
    build a docstring
    '''
    line = {k:dict(loads(line))[k] for k in fields}
    return line
    

