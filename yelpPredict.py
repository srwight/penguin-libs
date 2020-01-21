from keras.models import load_model
from keras.preprocessing import text, sequence
from nltk.corpus import stopwords
import os, string
import numpy as np

os.chdir(r'C:\Users\user\Desktop\Revature\Projects\Yelp')

#region
No_of_Words = 5000
Max_Seq = 200
Embed_Dim = 100

model = load_model('SAmodel.h5')

tokenizer = text.Tokenizer(num_words = No_of_Words, filters = '"#&()*+,-./;:<=>?@[\]^_`{|}~', lower=True)

stop_words = set(stopwords.words('english')) #'if', 'and', 'the', etc.
#endregion

def preprocess(text):
    translation = str.maketrans('', '', string.punctuation)
    text = text.translate(translation)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def predict_review(text):
    text = preprocess(text)
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences([text])
    text = sequence.pad_sequences(text, maxlen = Max_Seq)
    pred = model.predict(text)
    pred = np.argmax(pred)+1
    preds = str(pred)
    if pred > 3:
        print("Wow, such service. Great business, " + preds + " Stars!")
    elif pred < 3:
        print("Wow, awful business. Very poor, " + preds + " Stars.")
    else:
        print("Very business. Okay. " + preds + " Stars.")

predict_review(input("Tell the computer about your experience: "))