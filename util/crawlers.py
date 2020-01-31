from json import loads as json_lds
from string import digits, punctuation
from contractions import fix as cfix

class baseCrawler():
    
    def _getKeys(self, keys:list, injson:str) -> dict:
        outstring = {key:dict(json_lds(injson))[key] for key in keys}
        return outstring

    def _preprocess(self, instring:str) -> str:
        instring = self._demotify(instring)
        instring = instring.lower()
        instring = cfix(instring)
        instring = instring.translate(str.maketrans('','',punctuation))
        instring = instring.translate(str.maketrans('','',digits))
        return instring

    def _getTokens(self, instring:str) -> list:
        return instring.split()
    
    def _demotify(self, indoc:str) -> str:
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
            '&'  :' and '
        }

        for key in emoticon_dict:
            indoc = indoc.replace(key, emoticon_dict[key])

        return indoc

class gatherVocabFreq(baseCrawler):
    def __init__(self,filename:str,callbackFunc=None, callbackfreq=10000, 
                debug=False, debugRows=10000):
        self._filename=filename
        self._callbackFunc=callbackFunc
        self._callbackFreq = callbackfreq
        self._debug=debug
        self._debugRows=debugRows
        self._vocab=dict()
        self._docs=0

    def _fit(self, instring:str):
        instring = self._getKeys(['text',], instring)['text']
        instring = self._preprocess(instring)
        tokens = set(self._getTokens(instring))
        for token in tokens:
            if token in self._vocab:
                self._vocab[token] += 1
            else:
                self._vocab[token] = 1
        
    def crawl(self):
        filein = self._filename
        if self._callbackFunc is not None:
            self._callbackFunc('Crawler starting.')
        with open(filein, 'rt') as fl_in:
            for i, line in enumerate(fl_in):
                self._docs = i
                if self._debug:
                    if i == self._debugRows: break
                self._fit(line)
                if self._callbackFunc is not None:
                    if i % self._callbackFreq == 0 and i > 0:
                        self._callbackFunc(f'\rCrawling line {i}.', end='')
        if self._callbackFunc is not None:
            self._callbackFunc(f'\nJob\'s Done! {self._docs} lines crawled!')
    
    def getStopWords(self, minnum:int=1,maxfreq:float=0.6,
                    alwayskeep:set={'not'}):
        stopwords = set()
        for word in self._vocab:
            if word in alwayskeep:
                continue
            if self._vocab[word] < minnum or self._vocab[word] / self._docs > maxfreq:
                stopwords.add(word)
        return(stopwords)

class gatherBigramFreq(baseCrawler):
    def __init__(self, filename:str, stopwords:set=set(), 
                callbackFunc=None,callbackFreq=10000,debug=False,debugrecords=10000,
                keys = ['text','stars']):
        self._filename = filename
        self._stopwords = stopwords
        self._callbackFunc = callbackFunc
        self._callbackFreq = callbackFreq
        self._debug = debug
        self._debugrecords = debugrecords
        self._keys = keys
        self._bigrams = dict()
        self._bigramsStars = dict()
        self._vocab = dict()
        self._docs = 0

    def _getBigrams(self, instring:str) -> list:
        tokens = self._getTokens(instring)
        bigrams = set()
        for i in range(len(tokens)-1):
            if tokens[i] in self._stopwords or tokens[i+1] in self._stopwords:
                continue
            bigrams.add(f'{tokens[i]} {tokens[i+1]}')
        return bigrams

    def _fit(self, instring:str, stars:float):
        instring = self._preprocess(instring)
        bigrams = self._getBigrams(instring)
        for bigram in bigrams:
            if bigram in self._bigrams:
                self._bigrams[bigram] += 1
                self._bigramsStars[bigram].append(stars)
            else:
                self._bigrams[bigram] = 1
                self._bigramsStars[bigram] = [stars,]

    def crawl(self):
        if self._callbackFunc is not None:
            self._callbackFunc('Crawler Starting')
        with open(self._filename, 'rt') as fl_in:
            for i, line in enumerate(fl_in):
                self._docs=i
                line = self._getKeys(self._keys, line)
                self._fit(line['text'],line['stars'])

                ## Callbacks
                if self._callbackFunc is not None:
                    if i % self._callbackFreq == 0 and i > 0:
                        self._callbackFunc(f'\rProcessing line {i}.', end='')
                
                ## Debug Quit
                if self._debug:
                    if i >= self._debugrecords:
                        break
            if self._callbackFunc is not None:
                self._callbackFunc(f'\nJob\'s done! {self._docs} lines crawled!')

    def getBigrams(self, minnum:int=2):
        bigrams=dict()
        for i, bigram in enumerate(self._bigrams):
            if self._bigrams[bigram] < minnum:
                continue
            bigrams[bigram] = self._bigrams[bigram]
        return bigrams

    
            
