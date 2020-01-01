import pymongo, urllib.parse, time

# Enter your username and password given to access the MongoDB
user = ''
password = ''
ip = ''

# Initializes client that will communicate with the db
client = pymongo.MongoClient('mongodb://%s:%s@%s' % (urllib.parse.quote_plus(user), urllib.parse.quote_plus(password), ip), port=2343)

# Define starting times for execution timing.
times = []
start = time.time()
prev = start

# Loop through 5 randomly selected samples, print its raw data, and append the time it took
for data in client.admin.datasets.aggregate([{'$sample': {'size': 5}}]):
    print(data)
    times.append(time.time())

# Display time it took for program to finish loading the 5 sample data pieces
print(f'Program finished loading 5 data entries in {time.time() - start:.2f}s total.')
for i, curTime in enumerate(times):
    print(f'Entry {i} took {curTime - prev:.2f}s.')
    prev = curTime

client.close()