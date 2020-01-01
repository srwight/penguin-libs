import pymongo, urllib.parse

def connect(user:str, passwd:str, ip:str) -> pymongo.MongoClient:
    """Initializes the client that will communicate with the db."""
    return pymongo.MongoClient('mongodb://%s:%s@%s' % (urllib.parse.quote_plus(user), urllib.parse.quote_plus(passwd), ip), port=2343)