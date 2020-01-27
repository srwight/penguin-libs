def preprocessing(text):
    ''' The full preprocessing file. '''
    from preprocess import preprocess
    from cleaning_data import clean_d
    from demoticon import demotify
    clean_d(text)
    demotify(text)
    preprocess(text)
    return text