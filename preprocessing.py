def preprocessing(text):
    ''' The full preprocessing file. '''
    from mlserv.models.preprocess import preprocess
    from mlserv.models.cleaning_data import clean_d
    from mlserv.models.demoticon import demotify
    clean_d(text)
    demotify(text)
    preprocess(text)
    return text